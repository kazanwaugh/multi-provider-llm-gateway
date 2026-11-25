# AI Prompt and Audit Log

## Prompt used for the initial draft

I supplied the instructor's Step 2 sample prompt to ChatGPT and worked through the generated files in stages. The prompt asked for a Python 3.11+ gateway with Pydantic v2 types, official asynchronous OpenAI and Anthropic clients, and a shared `stream(prompt, model, temperature)` interface. It specified normalized `StreamChunk` events, provider-reported token counts, TTFT, throughput, total latency, standard per-million-token USD pricing, and safe typed errors for authentication, rate limits, and connection failures. It asked for `schemas.py`, `pricing.py`, and `gateway.py`.

For this implementation, we used GPT-4o mini and Claude Haiku 4.5 because the Claude 3.5 examples in the prompt had been retired. The initial schema and pricing check and the first gateway check used simulated inputs. The audit below records what changed after checking the first draft.

## Gateway timing and telemetry audit

Simulated text arrived at 100 ms and 300 ms. Final usage arrived at 1.1 s.

| Finding | First draft | After the fix |
| --- | --- | --- |
| Missing live telemetry | First text chunk had no telemetry | First text chunk reports 100 ms TTFT |
| Usage report delay counted as generation time | 4.0 tokens/sec | 20.0 tokens/sec using first-to-last text arrival |

Total request time remains 1.1 s. These results use a fake clock and simulated provider events.

\`\`\`diff
--- gateway.py (first draft)
+++ gateway.py (after fixes)
@@ -112,6 +112,7 @@
         )
         started = perf_counter()
         first_text_at: float | None = None
+        last_text_at: float | None = None
         prompt_tokens: int | None = None
         completion_tokens: int | None = None

@@ -122,10 +123,18 @@
                 if output_count is not None:
                     completion_tokens = output_count
                 if text:
+                    last_text_at = perf_counter()
                     if first_text_at is None:
-                        first_text_at = perf_counter()
+                        first_text_at = last_text_at
                     yield StreamChunk(
-                        delta_text=text, provider=provider, model=model
+                        delta_text=text,
+                        provider=provider,
+                        model=model,
+                        telemetry=ChunkTelemetry(
+                            prompt_tokens=prompt_tokens,
+                            completion_tokens=completion_tokens,
+                            ttft_ms=(first_text_at - started) * 1000,
+                        ),
                     )
         except (openai.AuthenticationError, anthropic.AuthenticationError):
             raise AuthenticationError(f"{provider} rejected the API key.") from None
@@ -144,9 +153,9 @@

         ttft_ms = None
         tokens_per_second = None
-        if first_text_at is not None:
+        if first_text_at is not None and last_text_at is not None:
             ttft_ms = (first_text_at - started) * 1000
-            generation_seconds = finished - first_text_at
+            generation_seconds = last_text_at - first_text_at
             if completion_tokens > 1 and generation_seconds > 0:
                 tokens_per_second = (completion_tokens - 1) / generation_seconds

\`\`\`
