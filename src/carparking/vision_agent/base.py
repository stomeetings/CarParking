"""Vision agent, step 1: the data contracts and the abstract base.

The agent only PROPOSES what it sees (painted separator lines as image points). Python derives the layout parameters from those points on
the ground plane and validates them (later steps); `layout_extra.layout_from_dots` draws the bays. Nothing here calls a model or the network.

Coordinates of a `RawVisionProposal` are in the crop the model was shown, as fractions of its width / height (0 = left / top, 1 = right /
bottom); the caller maps them back to the full frame. `SlotCalibrationParameters` is already in full-frame fractions (what the layout
engine takes). All models are strict (no type coercion: "5" is not 5), frozen and refuse unknown fields.

Strict mode note: parse a model's JSON reply with `RawVisionProposal.model_validate_json(text)` (arrays become tuples there). From a
Python dict, tuples must be real tuples, or pass `strict=False` to `model_validate` deliberately.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from carparking.layout import MAX_STALL_DEG, MIN_STALL_DEG

MAX_SLOTS = 60  # the most bays one lane / zone may be asked for (a sanity bound on the model's count)
MAX_LINES = 100  # the most separator lines one proposal may carry

# a point in the crop: (x, y) fractions of its width / height, finite numbers in 0..1
Coord = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Point = tuple[Coord, Coord]


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, str_strip_whitespace=True)


class Visibility(str, Enum):
    """How well the painted line can be seen; the model must say so instead of inventing a line under a car."""
    CLEAR = "clear"
    FADED = "faded"
    OCCLUDED = "occluded"


class LineEndpoint(_Strict):
    """One separator line between two bays: its near (front) end and its far (back) end, in the crop."""
    front: Point
    back: Point
    visible: Visibility

    @model_validator(mode="after")
    def _has_length(self) -> "LineEndpoint":
        if self.front == self.back:
            raise ValueError("a line's front and back points must differ")
        return self


class RawVisionProposal(_Strict):
    """What the model saw: the separator lines (in order along the row), its own count and its confidence. Only observations; the
    angle, spacing and count that the layout uses are derived and checked in Python, the model's count is a cross-check."""
    lines: Annotated[list[LineEndpoint], Field(max_length=MAX_LINES)]
    estimated_count: Optional[Annotated[int, Field(ge=0, le=MAX_SLOTS)]]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    notes: str = Field(default="", max_length=2000)


class SlotCalibrationParameters(_Strict):
    """The derived, validated parameters the geometry engine takes (`layout_extra.layout_from_dots`): the row's first separator line as
    two full-frame points (front = near end, back = far end), the bay angle in degrees on the ground plane (90 = square), the number of
    bays and the name prefix ("A" -> A1, A2 ...)."""
    stall_angle: Annotated[float, Field(ge=MIN_STALL_DEG, le=MAX_STALL_DEG, allow_inf_nan=False)]
    line0_front_dot: Point
    line0_back_dot: Point
    total_slots: Annotated[int, Field(ge=1, le=MAX_SLOTS)]
    slot_prefix: Annotated[str, Field(min_length=1, max_length=8, pattern=r"^[A-Za-z0-9_-]+$")]

    @model_validator(mode="after")
    def _dots_differ(self) -> "SlotCalibrationParameters":
        if self.line0_front_dot == self.line0_back_dot:
            raise ValueError("line0_front_dot and line0_back_dot must differ")
        return self


class BaseVisionAgent(ABC):
    """A provider of line proposals (a vision model behind some API, a fake in tests). Swap providers by subclassing; the rest of the
    pipeline only sees `RawVisionProposal`."""

    @abstractmethod
    def propose(self, crop_png_path: str, constraints: dict) -> RawVisionProposal:
        """Look at the cropped picture of one lane / zone and return the separator lines seen. `constraints`: what the operator knows
        (e.g. {"rows": 1, "min_count": 4, "max_count": 12}). Raise on a provider error; return an empty `lines` list when none are seen."""
