import asyncio
import json
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from time import perf_counter

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .gateway import GatewayProviderError, UnifiedLLMGateway
from .pricing import PRICING, calculate_cost

SRC_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(SRC_DIR.parent / ".env", override=False)
    app.state.gateway = UnifiedLLMGateway()
    try:
        yield
    finally:
        await app.state.gateway.close()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=8000)
    temperature: float = Field(default=0.7, ge=0, le=1)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/")
async def index():
    return FileResponse(SRC_DIR / "index.html")


@app.get("/models")
async def models():
    return [
        {"provider": provider, "model": model}
        for provider, model in PRICING
    ]


@app.post("/generate")
async def generate(body: GenerateRequest, request: Request):
    async def stream():
        gateway = request.app.state.gateway
        response_parts: list[str] = []
        first_at = None
        last_update = float("-inf")

        try:
            encoding = await asyncio.to_thread(tiktoken.encoding_for_model, "gpt-4o-mini")
            input_estimate = len(encoding.encode_ordinary(body.prompt))

            async with aclosing(gateway.stream(body.prompt, body.model, body.temperature)) as chunks:
                async for chunk in chunks:
                    payload = chunk.model_dump(mode="json")

                    if not chunk.is_final and chunk.delta_text:
                        response_parts.append(chunk.delta_text)
                        now = perf_counter()
                        if first_at is None:
                            first_at = now

                        if now - last_update >= 0.2:
                            last_update = now
                            output_estimate = len(
                                encoding.encode_ordinary("".join(response_parts))
                            )
                            metrics = payload["telemetry"] or {}
                            prompt_count = metrics.get("prompt_tokens")
                            if prompt_count is None:
                                prompt_count = input_estimate
                            elapsed = now - first_at
                            metrics.update(
                                prompt_tokens=prompt_count,
                                completion_tokens=output_estimate,
                                tokens_per_second=(
                                    (output_estimate - 1) / elapsed
                                    if elapsed > 0 and output_estimate > 1
                                    else None
                                ),
                                total_cost_usd=str(calculate_cost(
                                    chunk.provider, chunk.model, prompt_count, output_estimate
                                )),
                            )
                            payload["telemetry"] = metrics
                        else:
                            payload["telemetry"] = None

                    yield sse(payload)
        except GatewayProviderError as error:
            yield sse({"error": str(error)})
        except Exception:
            yield sse({"error": "The application could not complete the request."})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
