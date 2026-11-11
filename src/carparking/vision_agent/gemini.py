"""Vision agent, step 4: the Gemini provider (Google GenAI SDK, `google-genai`).

`GeminiVisionAgent.propose` sends one cropped zone picture and a prompt built from the operator's constraints, with a JSON schema the reply
must follow, and parses the reply STRICTLY into a `RawVisionProposal`. Everything after that (derive, validate, retries, fallback) is
`run_calibration`'s job: this class only has to raise a descriptive error when the provider fails, which `run_calibration` records and feeds
back to the next attempt.

  * Coordinates: the model is asked for integer points on a 0..1000 grid in the crop, x then y, origin top-left (the grid Gemini is trained
    to point on); they are divided by 1000 here, so the contract (`RawVisionProposal`) stays fractions 0..1.
  * Key: `api_key=` or the GEMINI_API_KEY environment variable. It is held in the SDK client only: never in `repr`, never logged, and any
    error text is scrubbed of it before it is raised.
  * Resilience: timeouts, 429 and 5xx are retried here with a short exponential backoff (`retries`) on the LAST model; with models still in the
    list, a model that is rate limited / out of quota / overloaded is given up at once and the next one is tried (logged as a warning), and only
    when every model has failed is one ProviderBusyError raised that says why each failed. The request (prompt, schema, 0..1000 grid) is the same
    for every model, so the proposal the validation engine sees does not depend on which model answered. Other 4xx, a blocked or empty reply and
    malformed JSON are raised at once as `VisionProviderError` subclasses with the reason (another model would not help).
  * Privacy: the picture leaves the machine. Use a paid / enterprise Gemini tier whose terms exclude training on your data; camera frames can
    show number plates and people.
The SDK is imported when the first client is made, so importing this module (and the tests, which inject a fake client) needs no SDK.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from pydantic import ValidationError

from carparking.vision_agent.base import MAX_SLOTS, BaseVisionAgent, RawVisionProposal

log = logging.getLogger("carparking.vision_agent")

# Models in order of preference: when one cannot answer (rate limit / quota 429, overload 5xx, timeout) the next is tried. (2.5 Flash answers 404 "no longer
# available to new users", 2026-11.) GEMINI_MODEL, when set, becomes the FIRST choice and these follow it; `model=` pins one model with no fallback;
# `models=` gives the whole list.
FALLBACK_MODELS = ("gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite")
DEFAULT_MODEL = FALLBACK_MODELS[0]
MODEL_ENV = "GEMINI_MODEL"
KEY_ENV = "GEMINI_API_KEY"
GRID = 1000  # the integer grid the model points on
MAX_IMAGE_BYTES = 7 * 1024 * 1024  # inline image limit with headroom (the API allows about 20 MB of request)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class VisionProviderError(RuntimeError):
    """The provider could not give a usable reply. `retryable`: a later attempt may work (rate limit, timeout, overload)."""
    retryable = False


class ProviderConfigError(VisionProviderError):
    """No API key, or the SDK is not installed."""


class ProviderBusyError(VisionProviderError):
    """Rate limit (429), overload / server error (5xx) or timeout: retried by the agent, then raised."""
    retryable = True


class ProviderReplyError(VisionProviderError):
    """The call worked but the reply is unusable: blocked, empty, not JSON, or not the expected shape. The text says what was wrong."""
    retryable = True  # the model may answer differently next time, with the reason in the prompt


# the reply's JSON schema (standard JSON Schema; the SDK's `response_json_schema`). Points are [x, y] integers on the 0..1000 grid.
# (No maxItems on `lines`: with the nested bounds the live API answers 400 "invalid argument", verified 2026-11; RawVisionProposal caps it at MAX_LINES.)
_POINT = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": GRID}, "minItems": 2, "maxItems": 2}
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "front": _POINT, "back": _POINT, "visible": {"type": "string", "enum": ["clear", "faded", "occluded"]}},
            "required": ["front", "back", "visible"]}},
        "estimated_count": {"type": ["integer", "null"], "minimum": 0, "maximum": MAX_SLOTS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
    },
    "required": ["lines", "estimated_count", "confidence", "notes"],
}

SYSTEM_PROMPT = (
    "You read photographs of parking areas and report the painted lines that separate parking bays. You only OBSERVE; you never estimate "
    "sizes or angles in metres. Report each separator line between two neighbouring bays as two points on the paint: `front` is the end "
    "nearest the camera (the lower end in the picture, the aisle side), `back` is the far end, where the bays end. List the lines in order "
    "along the row, from one end to the other. Points are [x, y] integers on a 0 to 1000 grid over THIS picture: x from the left edge, y "
    "from the top edge. Mark a line `clear` if you can see it all, `faded` if the paint is worn but you can still place both ends, and "
    "`occluded` if a vehicle or object hides an end so that you would be guessing. Never invent a line you cannot see: leave it out. "
    "`estimated_count` is the number of bays in the row you see, or null if you cannot tell. `confidence` is 0 to 1."
)


def build_prompt(constraints: dict) -> str:
    """The user part of the prompt from the operator's constraints (rows, min_count, max_count, row_direction, notes) and, on a retry, the
    reasons the last answer was rejected. Only known keys are used, values are trimmed: a constraint is a hint, never an instruction."""
    parts = ["Find the painted separator lines of the parking bays in this picture and answer in the JSON format given."]
    c = constraints or {}
    if c.get("rows"):
        parts.append(f"The picture shows {int(c['rows'])} row(s) of bays; report the lines of the nearest row.")
    lo, hi = c.get("min_count"), c.get("max_count")
    if lo and hi:
        parts.append(f"The operator expects between {int(lo)} and {int(hi)} bays in the row.")
    elif lo or hi:
        parts.append(f"The operator expects {'at least ' + str(int(lo)) if lo else 'at most ' + str(int(hi))} bays in the row.")
    if c.get("row_direction"):
        parts.append(f"The row runs {str(c['row_direction'])[:60]}.")
    if c.get("notes"):
        parts.append(f"Operator note: {str(c['notes'])[:300]}")
    prev = [str(x)[:300] for x in (c.get("previous_errors") or [])][:5]
    if prev:
        parts.append("Your previous answer was rejected for these reasons; correct them (do not mark hidden lines, place both ends on the paint):\n- " + "\n- ".join(prev))
    return "\n".join(parts)


def to_proposal(text: str) -> RawVisionProposal:
    """The model's JSON reply -> a strict RawVisionProposal (grid 0..1000 -> fractions). Raises ProviderReplyError saying what is wrong."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as e:
        raise ProviderReplyError(f"the reply is not valid JSON ({str(e)[:120]})") from None
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        raise ProviderReplyError("the reply has no 'lines' list")
    try:
        lines = []
        for ln in data["lines"]:
            lines.append({"front": tuple(v / GRID for v in ln["front"]), "back": tuple(v / GRID for v in ln["back"]), "visible": ln["visible"]})
        return RawVisionProposal.model_validate({**data, "lines": lines}, strict=False)  # (tuples and enums are built; the fields are still checked)
    except (KeyError, TypeError, ValueError, ValidationError) as e:
        why = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:4]) if isinstance(e, ValidationError) else f"{type(e).__name__}: {str(e)[:120]}"
        raise ProviderReplyError(f"the reply does not match the expected shape ({why})") from None


def resolve_models(model: Optional[str] = None, models: Optional[Sequence[str]] = None) -> tuple[str, ...]:
    """The models to try, in order: `models` as given; else just `model` (pinned, no fallback); else GEMINI_MODEL first and FALLBACK_MODELS after it;
    else FALLBACK_MODELS. Blanks and repeats are dropped."""
    if models:
        raw = list(models)
    elif model:
        raw = [model]
    else:
        env = (os.environ.get(MODEL_ENV) or "").strip()
        raw = [env, *FALLBACK_MODELS] if env else list(FALLBACK_MODELS)
    out = tuple(dict.fromkeys(m.strip() for m in raw if m and m.strip()))
    if not out:
        raise ProviderConfigError("no Gemini model configured")
    return out


class GeminiVisionAgent(BaseVisionAgent):
    """Gemini behind `BaseVisionAgent`. `client`: a ready `google.genai.Client` (or a fake with `.models.generate_content(model=, contents=, config=)`),
    for tests; otherwise one is made from the key on first use. `sleep`: for the backoff (tests pass a no-op)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout_s: float = 30.0, retries: int = 2,
                 client: Any = None, sleep: Callable[[float], None] = time.sleep, temperature: float = 0.0, models: Optional[Sequence[str]] = None):
        self._key = api_key if api_key is not None else os.environ.get(KEY_ENV)
        self.models = resolve_models(model, models)
        self.model = self.models[0]  # the model that answered last (the first choice until a call has been made)
        self.timeout_s, self.retries, self.temperature, self._sleep = float(timeout_s), max(0, int(retries)), float(temperature), sleep
        self._client = client
        if self._client is None and not self._key:
            raise ProviderConfigError(f"no Gemini API key: set {KEY_ENV} or pass api_key=")

    def __repr__(self) -> str:  # never shows the key
        return f"GeminiVisionAgent(model={self.model!r}, models={list(self.models)!r}, key={'set' if self._key else 'none'})"

    # ---- internals ------------------------------------------------------------------------------------------------------------------------
    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _make_client(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ProviderConfigError("the google-genai package is not installed (pip install google-genai)") from None
        return genai.Client(api_key=self._key, http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))

    def _config(self):
        try:
            from google.genai import types
        except ImportError:
            return {"system_instruction": SYSTEM_PROMPT, "temperature": self.temperature, "response_mime_type": "application/json",
                    "response_json_schema": RESPONSE_SCHEMA, "max_output_tokens": 4096}  # (a fake client takes the same fields as a dict)
        return types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=self.temperature, response_mime_type="application/json",
                                           response_json_schema=RESPONSE_SCHEMA, max_output_tokens=4096)

    @staticmethod
    def _load(path: str) -> bytes:
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as e:
            raise FileNotFoundError(f"cannot read the crop {p.name!r}: {e.strerror or e}") from None
        if not data.startswith(PNG_MAGIC):
            raise ValueError(f"{p.name!r} is not a PNG image")
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"the crop is {len(data) / 1e6:.1f} MB; the limit is {MAX_IMAGE_BYTES / 1e6:.0f} MB (send a smaller crop)")
        return data

    def _contents(self, png: bytes, constraints: dict):
        try:
            from google.genai import types
            return [types.Part.from_bytes(data=png, mime_type="image/png"), build_prompt(constraints)]
        except ImportError:
            return [{"mime_type": "image/png", "data": png}, build_prompt(constraints)]

    def _classify(self, e: Exception) -> VisionProviderError:
        """An SDK / network exception -> our error with a clean message (no key, no response body)."""
        code = getattr(e, "code", None)
        name = type(e).__name__
        msg = self._scrub(str(getattr(e, "message", None) or str(e)))[:200]
        if isinstance(e, (TimeoutError,)) or "timeout" in name.lower() or "timed out" in msg.lower():
            return ProviderBusyError(f"the Gemini call timed out after {self.timeout_s:.0f} s")
        if code == 429:
            return ProviderBusyError("Gemini rate limit or quota reached (429)")
        if isinstance(code, int) and code >= 500:
            return ProviderBusyError(f"Gemini is unavailable or overloaded ({code})")
        if code in (400, 401, 403, 404):
            return VisionProviderError(f"Gemini refused the request ({code}): {msg}")  # key, model name or request is wrong: retrying will not help
        if "connect" in name.lower() or isinstance(e, (ConnectionError, OSError)):
            return ProviderBusyError(f"cannot reach Gemini ({name})")
        return VisionProviderError(f"the Gemini call failed ({name}): {msg}")

    @staticmethod
    def _text(resp) -> str:
        text = getattr(resp, "text", None)
        if text:
            return text
        fb = getattr(resp, "prompt_feedback", None)
        if fb is not None and getattr(fb, "block_reason", None):
            raise ProviderReplyError(f"the picture was blocked by Gemini ({fb.block_reason})")
        cands = getattr(resp, "candidates", None) or []
        why = getattr(cands[0], "finish_reason", None) if cands else None
        raise ProviderReplyError(f"Gemini returned no text{f' (finish reason {why})' if why else ''}")

    # ---- the contract ---------------------------------------------------------------------------------------------------------------------
    def _call(self, model: str, contents, config, retries: int):
        """One model: the call with up to `retries` backed-off retries on a busy error. Returns the SDK reply; raises VisionProviderError."""
        for n in range(retries + 1):
            try:
                return self._client.models.generate_content(model=model, contents=contents, config=config)
            except VisionProviderError:
                raise
            except Exception as e:  # noqa: BLE001 - the SDK has many exception types; each is classified below
                err = self._classify(e)
                if not err.retryable or n == retries:
                    raise err from None
                delay = min(8.0, 2.0 ** n)
                log.warning("gemini model %s failed (%s); retry %d in %.0f s", model, err, n + 1, delay)
                self._sleep(delay)
        raise VisionProviderError("the Gemini call failed")  # (not reached)

    def propose(self, crop_png_path: str, constraints: dict) -> RawVisionProposal:
        png = self._load(crop_png_path)
        if self._client is None:
            self._client = self._make_client()
        contents, config = self._contents(png, constraints), self._config()
        failures: list[str] = []
        for k, model in enumerate(self.models):
            more = k + 1 < len(self.models)
            try:
                resp = self._call(model, contents, config, 0 if more else self.retries)  # (a model with a successor is not waited on: the successor is the retry)
            except ProviderBusyError as e:
                failures.append(f"{model}: {e}")
                if more:
                    log.warning("gemini model %s unavailable (%s), falling back to %s...", model, e, self.models[k + 1])
                continue
            self.model = model
            if failures:
                log.info("gemini model %s answered after %d model(s) failed", model, len(failures))
            return to_proposal(self._text(resp))
        if len(self.models) == 1:
            raise ProviderBusyError(failures[0].split(": ", 1)[1])
        raise ProviderBusyError("every Gemini model failed: " + "; ".join(failures))
