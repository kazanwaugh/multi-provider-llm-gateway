from decimal import Decimal
from types import SimpleNamespace as Event

import pytest
import tiktoken

from src.gateway import UnifiedLLMGateway
from src.schemas import StreamChunk


class FakeStream:
    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def __aiter__(self):
        for event in self.events:
            yield event


def openai_chunk(text):
    return Event(choices=[Event(delta=Event(content=text))], usage=None)


def anthropic_text(text):
    return Event(
        type="content_block_delta",
        delta=Event(type="text_delta", text=text),
    )


def anthropic_usage(count):
    return Event(type="message_delta", usage=Event(output_tokens=count))


def build_fake_gateway(provider):
    gateway = UnifiedLLMGateway()
    if provider == "openai":
        events = [
            openai_chunk(None),
            openai_chunk("Hello"),
            openai_chunk(" world"),
            Event(choices=[], usage=Event(prompt_tokens=50, completion_tokens=5)),
        ]

        async def create(**kwargs):
            assert kwargs["stream"] is True
            assert kwargs["stream_options"]["include_usage"] is True
            return FakeStream(events)

        gateway._openai = Event(chat=Event(completions=Event(create=create)))
    else:
        events = [
            Event(type="message_start", message=Event(usage=Event(input_tokens=50))),
            anthropic_text("Hello"),
            anthropic_text(" world"),
            anthropic_usage(3),
            anthropic_usage(5),
            Event(type="message_stop"),
        ]
        gateway._anthropic = Event(
            messages=Event(stream=lambda **kwargs: FakeStream(events))
        )

    return gateway


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,model,expected_cost",
    [
        ("openai", "gpt-4o-mini", Decimal("0.000011")),
        ("anthropic", "claude-haiku-4-5-20251001", Decimal("0.000075")),
    ],
)
async def test_unified_stream_and_final_telemetry(
    provider, model, expected_cost, monkeypatch
):
    times = iter([10.0, 10.1, 10.3, 11.1])
    monkeypatch.setattr("src.gateway.perf_counter", lambda: next(times))
    gateway = build_fake_gateway(provider)
    chunks = [chunk async for chunk in gateway.stream(" hello" * 50, model)]

    assert all(isinstance(chunk, StreamChunk) for chunk in chunks)
    assert all(chunk.provider == provider for chunk in chunks)
    assert "".join(chunk.delta_text for chunk in chunks) == "Hello world"
    assert [chunk.is_final for chunk in chunks] == [False, False, True]
    assert chunks[0].telemetry.ttft_ms == pytest.approx(100.0)

    final = chunks[-1].telemetry
    assert final.prompt_tokens == 50
    assert final.completion_tokens == 5
    assert final.total_cost_usd == expected_cost
    assert final.tokens_per_second == pytest.approx(20.0)
    assert final.total_latency_seconds == pytest.approx(1.1)


@pytest.mark.asyncio
async def test_openai_prompt_tokens_match_local_tokenizer():
    model = "gpt-4o-mini"
    prompt = " hello" * 50
    encoding = tiktoken.encoding_for_model(model)
    local_count = len(encoding.encode(prompt))

    gateway = build_fake_gateway("openai")
    chunks = [chunk async for chunk in gateway.stream(prompt, model)]
    reported_count = chunks[-1].telemetry.prompt_tokens

    assert local_count > 0
    assert abs(reported_count - local_count) / local_count <= 0.10
