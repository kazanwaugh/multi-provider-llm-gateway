from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


Provider = Literal["openai", "anthropic"]


class ChunkTelemetry(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    total_latency_seconds: float | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)
    total_cost_usd: Decimal | None = Field(default=None, ge=0)


class StreamChunk(BaseModel):
    delta_text: str = ""
    provider: Provider
    model: str
    is_final: bool = False
    telemetry: ChunkTelemetry | None = None
