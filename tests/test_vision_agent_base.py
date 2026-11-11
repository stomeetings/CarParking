import pytest
from pydantic import ValidationError

from carparking.vision_agent.base import BaseVisionAgent, LineEndpoint, RawVisionProposal, SlotCalibrationParameters, Visibility

LINE = {"front": [0.2, 0.9], "back": [0.3, 0.4], "visible": "clear"}


def test_proposal_parses_model_json_and_is_strict():
    p = RawVisionProposal.model_validate_json('{"lines": [%s], "estimated_count": null, "confidence": 0.8, "notes": ""}' % __import__("json").dumps(LINE))
    assert p.lines[0].visible is Visibility.CLEAR and p.lines[0].front == (0.2, 0.9) and p.estimated_count is None
    bad = ['{"lines": [], "estimated_count": "5", "confidence": 0.5, "notes": ""}',  # no coercion of "5"
           '{"lines": [], "estimated_count": 3, "confidence": 1.5, "notes": ""}',
           '{"lines": [], "estimated_count": 3, "confidence": 0.5, "notes": "", "extra": 1}',
           '{"lines": [{"front": [0.1, 0.1], "back": [0.1, 0.1], "visible": "clear"}], "estimated_count": 1, "confidence": 0.5, "notes": ""}',
           '{"lines": [{"front": [2, 0.1], "back": [0.1, 0.5], "visible": "clear"}], "estimated_count": 1, "confidence": 0.5, "notes": ""}',
           '{"lines": [{"front": [0.1, 0.1], "back": [0.2, 0.5], "visible": "gone"}], "estimated_count": 1, "confidence": 0.5, "notes": ""}']
    for text in bad:
        with pytest.raises(ValidationError):
            RawVisionProposal.model_validate_json(text)
    with pytest.raises(ValidationError):
        p.confidence = 0.1  # frozen


def test_parameters_bounds():
    ok = dict(stall_angle=60.0, line0_front_dot=(0.2, 0.9), line0_back_dot=(0.3, 0.4), total_slots=7, slot_prefix="A")
    assert SlotCalibrationParameters(**ok).total_slots == 7
    for k, v in (("stall_angle", 10.0), ("stall_angle", 170.0), ("total_slots", 0), ("total_slots", 99), ("slot_prefix", ""), ("slot_prefix", "A 1"),
                 ("line0_back_dot", (0.2, 0.9))):
        with pytest.raises(ValidationError):
            SlotCalibrationParameters(**{**ok, k: v})


def test_base_is_abstract_and_a_subclass_works():
    with pytest.raises(TypeError):
        BaseVisionAgent()

    class Fake(BaseVisionAgent):
        def propose(self, crop_png_path, constraints):
            return RawVisionProposal(lines=[LineEndpoint(front=(0.1, 0.9), back=(0.2, 0.4), visible=Visibility.FADED)], estimated_count=1, confidence=0.5)

    assert Fake().propose("x.png", {}).lines[0].visible is Visibility.FADED
