import asyncio
import time

import pytest

import mlx_server.managers.generative as gm
from mlx_server.managers.generative import GenerativeModelManager


class _TokenizerWithThinking:
    def __init__(self):
        self.called_with = None

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None
    ):
        self.called_with = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        return "PROMPT"


class _TokenizerWithoutThinking:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT_NO_THINKING"


def test_build_prompt_passes_enable_thinking_when_supported():
    tok = _TokenizerWithThinking()
    prompt = GenerativeModelManager._build_prompt(
        tok,
        [{"role": "user", "content": "hi"}],
        enable_thinking=False,
    )
    assert prompt == "PROMPT"
    assert tok.called_with["enable_thinking"] is False


def test_build_prompt_ignores_enable_thinking_when_unsupported():
    tok = _TokenizerWithoutThinking()
    prompt = GenerativeModelManager._build_prompt(
        tok,
        [{"role": "user", "content": "hi"}],
        enable_thinking=False,
    )
    assert prompt == "PROMPT_NO_THINKING"


class _TokenizerVarKwargs:
    """apply_chat_template accepts **kwargs — the probe must detect support via
    the VAR_KEYWORD parameter, not just a named enable_thinking arg."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.calls.append(kwargs)
        return "PROMPT_VARKW"


class _TokenizerRejectsThinking:
    """Signature exposes **kwargs (so the probe thinks it's supported) but the
    template raises TypeError when enable_thinking is actually passed."""

    def __init__(self):
        self.call_count = 0

    def apply_chat_template(self, messages, **kwargs):
        self.call_count += 1
        if "enable_thinking" in kwargs:
            raise TypeError("apply_chat_template() got an unexpected keyword 'enable_thinking'")
        return "PROMPT_FALLBACK"


def test_build_prompt_detects_support_via_var_keyword():
    tok = _TokenizerVarKwargs()
    prompt = GenerativeModelManager._build_prompt(
        tok, [{"role": "user", "content": "hi"}], enable_thinking=True
    )
    assert prompt == "PROMPT_VARKW"
    assert tok.calls[0].get("enable_thinking") is True


def test_build_prompt_retries_without_enable_thinking_on_typeerror():
    """Regression (n39): the probe can be fooled; a template that rejects
    enable_thinking must fall back instead of 500-ing the request."""
    tok = _TokenizerRejectsThinking()
    prompt = GenerativeModelManager._build_prompt(
        tok, [{"role": "user", "content": "hi"}], enable_thinking=True
    )
    assert prompt == "PROMPT_FALLBACK"
    assert tok.call_count == 2  # first attempt raised, retry succeeded


# ---------------------------------------------------------------------------
# Eviction / unload safety against in-flight generations
# ---------------------------------------------------------------------------


def test_unload_refuses_in_use_model_unless_forced():
    mgr = GenerativeModelManager()
    mgr._models["m"] = (object(), object())
    mgr._in_use["m"] = 1

    assert mgr.unload("m") is False
    assert "m" in mgr._models

    assert mgr.unload("m", force=True) is True
    assert "m" not in mgr._models


def test_evict_stale_skips_in_use_then_evicts_when_idle():
    mgr = GenerativeModelManager()
    mgr._models["m"] = (object(), object())
    mgr._last_used["m"] = time.monotonic() - 10_000  # very stale
    mgr._in_use["m"] = 1

    assert mgr.evict_stale(ttl=1) == []
    assert "m" in mgr._models

    mgr._in_use["m"] = 0
    assert mgr.evict_stale(ttl=1) == ["m"]
    assert "m" not in mgr._models


# ---------------------------------------------------------------------------
# Async bridge + KV forwarding
# ---------------------------------------------------------------------------


async def test_bridge_disconnect_does_not_wedge_executor():
    """Regression (n1/n20): a slow client that disconnects mid-stream while the
    bounded queue is full must not permanently wedge the single worker thread."""

    def infinite():
        i = 0
        while True:
            yield i
            i += 1

    agen = gm._bridge_to_async(infinite)
    assert await agen.__anext__() == 0
    # Let the worker fill the 16-slot queue and block on put (backpressure).
    await asyncio.sleep(0.1)
    # Simulate client disconnect mid-stream.
    await agen.aclose()

    # The single shared executor must now be free: a fresh generation completes.
    def finite():
        yield "a"
        yield "b"

    out = []

    async def run():
        async for item in gm._bridge_to_async(finite):
            out.append(item)

    await asyncio.wait_for(run(), timeout=10)
    assert out == ["a", "b"]


async def test_stream_forwards_kv_and_sampler_kwargs(monkeypatch):
    """Regression (n25): KV-cache controls must reach stream_generate."""
    mgr = GenerativeModelManager()
    captured: dict = {}

    async def fake_load(model_id, path):
        return object(), _TokenizerWithoutThinking()

    def fake_stream_generate(model, tokenizer, prompt, **kwargs):
        captured.update(kwargs)
        yield object()

    monkeypatch.setattr(mgr, "load_model", fake_load)
    monkeypatch.setattr(gm, "stream_generate", fake_stream_generate)

    chunks = []
    async for chunk in mgr.stream(
        "m", "/p", [{"role": "user", "content": "hi"}],
        max_tokens=64, max_kv_size=4096, kv_bits=8, kv_group_size=32, quantized_kv_start=128,
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert captured["max_tokens"] == 64
    assert captured["max_kv_size"] == 4096
    assert captured["kv_bits"] == 8
    assert captured["kv_group_size"] == 32
    assert captured["quantized_kv_start"] == 128
    assert "sampler" in captured
    # in-use counter is released after the stream completes
    assert mgr._in_use.get("m", 0) == 0


async def test_stream_aclose_releases_in_use(monkeypatch):
    """Regression: a client disconnect mid-stream (which the SSE handler turns
    into agen.aclose()) must release the in-use counter and tear down the
    bridge, not leak until GC."""
    mgr = GenerativeModelManager()

    async def fake_load(model_id, path):
        return object(), _TokenizerWithoutThinking()

    def fake_stream_generate(model, tokenizer, prompt, **kwargs):
        while True:
            yield object()

    monkeypatch.setattr(mgr, "load_model", fake_load)
    monkeypatch.setattr(gm, "stream_generate", fake_stream_generate)

    agen = mgr.stream("m", "/p", [{"role": "user", "content": "hi"}]).__aiter__()
    await agen.__anext__()
    assert mgr._in_use.get("m") == 1

    await agen.aclose()
    assert mgr._in_use.get("m", 0) == 0
