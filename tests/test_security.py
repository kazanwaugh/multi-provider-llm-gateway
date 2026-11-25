from types import SimpleNamespace

import httpx
import openai
import pytest

from src.gateway import AuthenticationError, UnifiedLLMGateway


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,key_name,bad_value",
    [
        ("gpt-4o-mini", "OPENAI_API_KEY", None),
        ("gpt-4o-mini", "OPENAI_API_KEY", "malformed-key"),
        ("claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY", None),
        ("claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY", "malformed-key"),
    ],
)
async def test_missing_or_malformed_key_is_reported(
    model, key_name, bad_value, monkeypatch
):
    if bad_value is None:
        monkeypatch.delenv(key_name, raising=False)
    else:
        monkeypatch.setenv(key_name, bad_value)

    with pytest.raises(AuthenticationError, match=key_name):
        async for _ in UnifiedLLMGateway().stream("Say hello", model):
            pass


@pytest.mark.asyncio
async def test_provider_error_does_not_expose_key(caplog):
    fake_secret = "sk-test-fake-key-for-redaction"
    response = httpx.Response(
        401, request=httpx.Request("POST", "https://example.invalid")
    )

    async def reject(**kwargs):
        raise openai.AuthenticationError(
            f"Provider rejected {fake_secret}", response=response, body=None
        )

    gateway = UnifiedLLMGateway()
    gateway._openai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=reject))
    )
    with pytest.raises(AuthenticationError) as error:
        async for _ in gateway.stream("Say hello", "gpt-4o-mini"):
            pass

    assert str(error.value) == "openai rejected the API key."
    assert fake_secret not in str(error.value)
    assert fake_secret not in repr(error.value)
    assert fake_secret not in caplog.text
