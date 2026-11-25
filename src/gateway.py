import os
from collections.abc import AsyncGenerator
from time import perf_counter

import anthropic
import openai

from .pricing import PRICING, calculate_cost
from .schemas import ChunkTelemetry, Provider, StreamChunk


class GatewayProviderError(Exception):
    """A safe error message that can be shown to the user."""


class AuthenticationError(GatewayProviderError):
    pass


class RateLimitExceededError(GatewayProviderError):
    pass


class ProviderConnectionError(GatewayProviderError):
    pass


ProviderEvent = tuple[str, int | None, int | None]
MODEL_PROVIDERS = {model: provider for provider, model in PRICING}


def _read_key(variable: str, prefix: str) -> str:
    key = os.getenv(variable, "")
    if not key.startswith(prefix) or len(key) < 16 or any(c.isspace() for c in key):
        raise AuthenticationError(
            f"Set a valid {variable} in your local environment."
        )
    return key


class UnifiedLLMGateway:
    def __init__(self) -> None:
        self._openai: openai.AsyncOpenAI | None = None
        self._anthropic: anthropic.AsyncAnthropic | None = None

    async def close(self) -> None:
        if self._openai is not None:
            await self._openai.close()
            self._openai = None
        if self._anthropic is not None:
            await self._anthropic.close()
            self._anthropic = None

    async def _openai_events(
        self, prompt: str, model: str, temperature: float
    ) -> AsyncGenerator[ProviderEvent, None]:
        if self._openai is None:
            self._openai = openai.AsyncOpenAI(
                api_key=_read_key("OPENAI_API_KEY", "sk-"), max_retries=0
            )
        response = await self._openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=512,
            stream=True,
            stream_options={"include_usage": True},
        )
        async with response:
            async for event in response:
                if event.choices:
                    text = event.choices[0].delta.content
                    if text:
                        yield text, None, None
                if event.usage is not None:
                    yield "", event.usage.prompt_tokens, event.usage.completion_tokens

    async def _anthropic_events(
        self, prompt: str, model: str, temperature: float
    ) -> AsyncGenerator[ProviderEvent, None]:
        if self._anthropic is None:
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=_read_key("ANTHROPIC_API_KEY", "sk-ant-"), max_retries=0
            )
        async with self._anthropic.messages.stream(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512,
        ) as response:
            async for event in response:
                if event.type == "message_start":
                    yield "", event.message.usage.input_tokens, None
                elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield event.delta.text, None, None
                elif event.type == "message_delta":
                    yield "", None, event.usage.output_tokens

    async def stream(
        self, prompt: str, model: str, temperature: float = 0.7
    ) -> AsyncGenerator[StreamChunk, None]:
        provider: Provider | None = MODEL_PROVIDERS.get(model)
        if provider is None:
            raise GatewayProviderError("Choose a supported model.")
        if not prompt.strip() or not 0 <= temperature <= 1:
            raise GatewayProviderError("Enter a prompt and a temperature from 0 to 1.")

        events = (
            self._openai_events(prompt, model, temperature)
            if provider == "openai"
            else self._anthropic_events(prompt, model, temperature)
        )
        started = perf_counter()
        first_text_at: float | None = None
        last_text_at: float | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            async for text, input_count, output_count in events:
                if input_count is not None:
                    prompt_tokens = input_count
                if output_count is not None:
                    completion_tokens = output_count
                if text:
                    last_text_at = perf_counter()
                    if first_text_at is None:
                        first_text_at = last_text_at
                    yield StreamChunk(
                        delta_text=text,
                        provider=provider,
                        model=model,
                        telemetry=ChunkTelemetry(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            ttft_ms=(first_text_at - started) * 1000,
                        ),
                    )
        except (openai.AuthenticationError, anthropic.AuthenticationError):
            raise AuthenticationError(f"{provider} rejected the API key.") from None
        except (openai.RateLimitError, anthropic.RateLimitError):
            raise RateLimitExceededError(f"{provider} rate limit reached.") from None
        except (openai.APIConnectionError, anthropic.APIConnectionError):
            raise ProviderConnectionError(f"Could not connect to {provider}.") from None
        except (openai.APIError, anthropic.APIError):
            raise GatewayProviderError(f"{provider} could not complete the request.") from None
        finally:
            await events.aclose()

        finished = perf_counter()
        if prompt_tokens is None or completion_tokens is None:
            raise GatewayProviderError(f"{provider} did not report final token usage.")

        ttft_ms = None
        tokens_per_second = None
        if first_text_at is not None and last_text_at is not None:
            ttft_ms = (first_text_at - started) * 1000
            generation_seconds = last_text_at - first_text_at
            if completion_tokens > 1 and generation_seconds > 0:
                tokens_per_second = (completion_tokens - 1) / generation_seconds

        yield StreamChunk(
            provider=provider,
            model=model,
            is_final=True,
            telemetry=ChunkTelemetry(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
                total_latency_seconds=finished - started,
                tokens_per_second=tokens_per_second,
                total_cost_usd=calculate_cost(
                    provider, model, prompt_tokens, completion_tokens
                ),
            ),
        )
