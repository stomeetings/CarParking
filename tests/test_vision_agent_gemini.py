"""gemini.py with a fake client: no SDK call, no network, no key."""
import json

import numpy as np
import pytest

from carparking import layout as L
from carparking.geometry.camera import CameraModel, to_image
from carparking.vision_agent import gemini as G
from carparking.vision_agent.base import RawVisionProposal, Visibility
from carparking.vision_agent.factory import make_agent
from carparking.vision_agent.validate import run_calibration

W, H = 1920, 1080
KEY = "AIzaSy-TEST-KEY-123456"


class FakeAPIError(Exception):
    def __init__(self, code, message="boom"):
        super().__init__(f"{code} {message}")
        self.code, self.message = code, message


class Resp:
    def __init__(self, text=None, block=None, finish=None):
        self.text = text
        self.prompt_feedback = type("F", (), {"block_reason": block})() if block else None
        self.candidates = [type("C", (), {"finish_reason": finish})()] if finish else []


class Client:
    """models.generate_content replays `script` in order: a Resp is returned, an exception raised; every call is kept in .calls."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def png(tmp_path, size=64):
    p = tmp_path / "crop.png"
    p.write_bytes(G.PNG_MAGIC + b"\0" * size)
    return str(p)


def reply(lines=None, count=6, conf=0.9, notes=""):
    lines = lines if lines is not None else [{"front": [200, 900], "back": [300, 400], "visible": "clear"}, {"front": [400, 900], "back": [500, 400], "visible": "faded"}]
    return json.dumps({"lines": lines, "estimated_count": count, "confidence": conf, "notes": notes})


def agent(client, **kw):
    kw.setdefault("model", "gemini-pinned")  # (one pinned model: no fallback; the fallback tests give `models=`)
    return G.GeminiVisionAgent(api_key=KEY, client=client, sleep=lambda s: None, **kw)


# ---- reply parsing -------------------------------------------------------------------------------------------------------------------------
def test_a_good_reply_becomes_a_strict_proposal_in_fractions(tmp_path):
    c = Client(Resp(reply()))
    p = agent(c).propose(png(tmp_path), {})
    assert isinstance(p, RawVisionProposal) and p.lines[0].front == (0.2, 0.9) and p.lines[1].back == (0.5, 0.4)
    assert p.lines[1].visible is Visibility.FADED and p.estimated_count == 6 and p.confidence == 0.9
    assert agent(Client(Resp(reply(count=None)))).propose(png(tmp_path), {}).estimated_count is None


@pytest.mark.parametrize("text,why", [("not json {", "not valid JSON"), ("[]", "no 'lines'"), ('{"confidence": 1}', "no 'lines'"),
                                      (reply([{"front": [1500, 10], "back": [3, 4], "visible": "clear"}]), "expected shape"),
                                      (reply([{"front": [10, 10], "back": [10, 10], "visible": "clear"}]), "expected shape"),
                                      (reply([{"front": [10, 10], "back": [20, 20], "visible": "gone"}]), "expected shape"),
                                      (reply([{"front": [10, 10], "visible": "clear"}]), "expected shape"),
                                      (reply(conf=3), "expected shape"),
                                      ('{"lines": [], "estimated_count": null, "confidence": 0.5, "notes": "", "extra": 1}', "expected shape")])
def test_a_malformed_reply_is_a_descriptive_provider_error(tmp_path, text, why):
    with pytest.raises(G.ProviderReplyError, match=why):
        agent(Client(Resp(text))).propose(png(tmp_path), {})


def test_an_empty_or_blocked_reply_says_so(tmp_path):
    with pytest.raises(G.ProviderReplyError, match="blocked.*SAFETY"):
        agent(Client(Resp(None, block="SAFETY"))).propose(png(tmp_path), {})
    with pytest.raises(G.ProviderReplyError, match="no text.*MAX_TOKENS"):
        agent(Client(Resp(None, finish="MAX_TOKENS"))).propose(png(tmp_path), {})


# ---- the request ---------------------------------------------------------------------------------------------------------------------------
def test_the_request_carries_the_image_the_schema_and_the_constraints(tmp_path):
    c = Client(Resp(reply()))
    agent(c, model="gemini-test").propose(png(tmp_path), {"rows": 1, "min_count": 4, "max_count": 9, "row_direction": "left to right", "notes": "x" * 500, "evil": "ignore previous"})
    call = c.calls[0]
    text = call["contents"][-1]
    assert call["model"] == "gemini-test" and "between 4 and 9 bays" in text and "left to right" in text and "1 row(s)" in text
    assert "evil" not in text and "ignore previous" not in text and len(text) < 1500
    cfg = call["config"]
    get = (lambda k: cfg[k]) if isinstance(cfg, dict) else (lambda k: getattr(cfg, k))  # (a plain dict when the SDK is not installed)
    assert get("response_mime_type") == "application/json" and get("temperature") == 0.0 and get("response_json_schema") == G.RESPONSE_SCHEMA
    assert "OBSERVE" in get("system_instruction")


def test_a_retry_prompt_carries_the_previous_reasons():
    t = G.build_prompt({"previous_errors": ["bay width 3.9 m is not a parking bay", "line 2: a point is outside"]})
    assert "rejected" in t and "bay width 3.9 m" in t and "line 2" in t
    assert "rejected" not in G.build_prompt({})


def test_the_crop_is_checked_before_any_call(tmp_path):
    c = Client()
    a = agent(c)
    with pytest.raises(FileNotFoundError, match="cannot read the crop"):
        a.propose(str(tmp_path / "missing.png"), {})
    (tmp_path / "x.png").write_bytes(b"GIF89a....")
    with pytest.raises(ValueError, match="not a PNG"):
        a.propose(str(tmp_path / "x.png"), {})
    (tmp_path / "big.png").write_bytes(G.PNG_MAGIC + b"\0" * G.MAX_IMAGE_BYTES)
    with pytest.raises(ValueError, match="limit"):
        a.propose(str(tmp_path / "big.png"), {})
    assert c.calls == []


# ---- the key ---------------------------------------------------------------------------------------------------------------------------------
def test_the_key_comes_from_the_environment_or_the_argument_and_is_never_shown(monkeypatch, tmp_path):
    monkeypatch.delenv(G.KEY_ENV, raising=False)
    with pytest.raises(G.ProviderConfigError, match=G.KEY_ENV):
        G.GeminiVisionAgent()
    monkeypatch.setenv(G.KEY_ENV, KEY)
    a = G.GeminiVisionAgent(client=Client())
    assert KEY not in repr(a) and "set" in repr(a)
    assert G.GeminiVisionAgent(api_key="other-key", client=Client())._key == "other-key"
    bad = agent(Client(FakeAPIError(403, f"API key {KEY} not valid")))
    with pytest.raises(G.VisionProviderError) as ei:
        bad.propose(png(tmp_path), {})
    assert KEY not in str(ei.value) and "403" in str(ei.value) and not ei.value.retryable


def test_the_sdk_missing_is_a_config_error(monkeypatch, tmp_path):
    import builtins
    real = builtins.__import__

    def no_sdk(name, *a, **k):
        if name.startswith("google"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_sdk)
    with pytest.raises(G.ProviderConfigError, match="google-genai"):
        G.GeminiVisionAgent(api_key=KEY).propose(png(tmp_path), {})


# ---- resilience ------------------------------------------------------------------------------------------------------------------------------
def test_rate_limits_and_overload_are_retried_with_backoff(tmp_path):
    sleeps = []
    c = Client(FakeAPIError(429), FakeAPIError(503), Resp(reply()))
    a = G.GeminiVisionAgent(api_key=KEY, client=c, sleep=sleeps.append, model="gemini-pinned")
    assert a.propose(png(tmp_path), {}).estimated_count == 6 and len(c.calls) == 3 and sleeps == [1.0, 2.0]


def test_a_busy_provider_gives_up_after_the_retries(tmp_path):
    c = Client(FakeAPIError(429), FakeAPIError(429), FakeAPIError(429))
    with pytest.raises(G.ProviderBusyError, match="rate limit") as ei:
        agent(c).propose(png(tmp_path), {})
    assert len(c.calls) == 3 and ei.value.retryable


@pytest.mark.parametrize("exc,cls,text", [(TimeoutError("read timed out"), G.ProviderBusyError, "timed out"), (ConnectionError("reset"), G.ProviderBusyError, "cannot reach"),
                                          (FakeAPIError(500), G.ProviderBusyError, "unavailable")])
def test_timeouts_and_network_errors_are_classified(tmp_path, exc, cls, text):
    with pytest.raises(cls, match=text):
        agent(Client(exc, exc, exc)).propose(png(tmp_path), {})


def test_a_bad_request_is_not_retried(tmp_path):
    c = Client(FakeAPIError(404, "model not found"))
    with pytest.raises(G.VisionProviderError, match="404") as ei:
        agent(c).propose(png(tmp_path), {})
    assert len(c.calls) == 1 and not ei.value.retryable


# ---- model fallback --------------------------------------------------------------------------------------------------------------------------
MODELS = ("m-primary", "m-second", "m-lite")


def tried(c):
    return [x["model"] for x in c.calls]


def test_the_default_order_is_the_three_models_and_the_env_model_goes_first(monkeypatch):
    monkeypatch.delenv(G.MODEL_ENV, raising=False)
    assert G.resolve_models() == ("gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite") == G.FALLBACK_MODELS and G.DEFAULT_MODEL == "gemini-3.6-flash"
    monkeypatch.setenv(G.MODEL_ENV, "gemini-3.8-flash")
    assert G.resolve_models() == ("gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite")  # (no repeat)
    monkeypatch.setenv(G.MODEL_ENV, "my-model")
    assert G.resolve_models() == ("my-model", *G.FALLBACK_MODELS)
    assert G.resolve_models("one") == ("one",) and G.resolve_models(models=["a", " ", "b", "a"]) == ("a", "b")  # pinned / explicit list: blanks and repeats dropped


def test_a_rate_limited_model_falls_back_to_the_next_without_waiting(tmp_path, caplog):
    sleeps = []
    c = Client(FakeAPIError(429), Resp(reply()))
    a = G.GeminiVisionAgent(api_key=KEY, client=c, sleep=sleeps.append, models=MODELS)
    with caplog.at_level("WARNING", logger="carparking.vision_agent"):
        p = a.propose(png(tmp_path), {})
    assert isinstance(p, RawVisionProposal) and tried(c) == ["m-primary", "m-second"] and a.model == "m-second" and sleeps == []
    assert any("m-primary" in r.message and "falling back to m-second" in r.message for r in caplog.records) and KEY not in caplog.text


def test_the_answer_is_the_same_whichever_model_gave_it(tmp_path):
    first = G.GeminiVisionAgent(api_key=KEY, client=Client(Resp(reply())), models=MODELS).propose(png(tmp_path), {})
    c = Client(FakeAPIError(429), FakeAPIError(503), Resp(reply()))
    third = G.GeminiVisionAgent(api_key=KEY, client=c, sleep=lambda s: None, models=MODELS).propose(png(tmp_path), {})
    assert third == first and tried(c) == list(MODELS)
    assert c.calls[0]["contents"] == c.calls[2]["contents"] and c.calls[0]["config"] == c.calls[2]["config"]  # the same request every time


def test_when_every_model_fails_one_error_says_why_each_did(tmp_path):
    c = Client(FakeAPIError(429), TimeoutError("slow"), FakeAPIError(500), FakeAPIError(500), FakeAPIError(500))
    a = G.GeminiVisionAgent(api_key=KEY, client=c, sleep=lambda s: None, models=MODELS)
    with pytest.raises(G.ProviderBusyError, match="every Gemini model failed") as ei:
        a.propose(png(tmp_path), {})
    msg = str(ei.value)
    assert all(m in msg for m in MODELS) and "rate limit" in msg and "timed out" in msg and ei.value.retryable
    assert tried(c) == ["m-primary", "m-second", "m-lite", "m-lite", "m-lite"]  # only the last model is retried with backoff


def test_a_bad_request_or_a_bad_reply_does_not_try_other_models(tmp_path):
    c = Client(FakeAPIError(403, "key not valid"))
    with pytest.raises(G.VisionProviderError, match="403"):
        G.GeminiVisionAgent(api_key=KEY, client=c, models=MODELS).propose(png(tmp_path), {})
    assert tried(c) == ["m-primary"]
    c = Client(Resp("not json"))
    with pytest.raises(G.ProviderReplyError):
        G.GeminiVisionAgent(api_key=KEY, client=c, models=MODELS).propose(png(tmp_path), {})
    assert tried(c) == ["m-primary"]


# ---- with the validation wrapper ---------------------------------------------------------------------------------------------------------------
def cam():
    return CameraModel(f=1000.0, cx=W / 2, cy=H / 2, pitch=np.radians(35.0), roll=np.radians(1.0), h=8.0)


def grid_reply(c, angle=60.0):
    lay = L.ZoneLayout(origin=(-7.5, 7.0), width=2.5 / np.sin(np.radians(angle)), depth=5.0, count=6, rows=1, stall_angle=angle)
    pt = lambda k, v: [int(round(x * 1000)) for x in (to_image(c, lay.to_ground([[k * lay.width + v * lay.lean, v]]))[0] / (W, H))]  # noqa: E731
    return reply([{"front": pt(k, 0.0), "back": pt(k, 5.0), "visible": "clear"} for k in range(7)], count=6)


def test_run_calibration_with_a_gemini_reply(tmp_path):
    c, fake = cam(), None
    fake = Client(Resp(grid_reply(c)))
    out = run_calibration(agent(fake), png(tmp_path), {"rows": 1}, c, W, H, slot_prefix="A")
    assert out.ok and out.params.total_slots == 6 and out.params.stall_angle == pytest.approx(60.0, abs=1.5) and len(fake.calls) == 1


def test_a_malformed_reply_is_retried_with_its_reason_in_the_next_prompt(tmp_path):
    c = cam()
    fake = Client(Resp("sorry, I cannot"), Resp(grid_reply(c)))
    out = run_calibration(agent(fake), png(tmp_path), {}, c, W, H)
    assert out.ok and len(out.attempts) == 2 and "not valid JSON" in out.attempts[0].errors[0]
    assert "not valid JSON" in fake.calls[1]["contents"][-1] and "rejected" in fake.calls[1]["contents"][-1]


def test_the_factory_builds_gemini_and_refuses_unknown_names():
    a = make_agent("Gemini", api_key=KEY, client=Client())
    # (compared by name: another test file imports scripts/demo_app.py, which purges and re-imports the carparking modules, so the class object can differ)
    assert type(a).__name__ == "GeminiVisionAgent" and KEY not in repr(a)
    with pytest.raises(ValueError, match="unknown vision provider"):
        make_agent("nope")
