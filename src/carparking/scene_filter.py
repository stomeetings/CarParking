"""Scene guardrail: decide from the detector's output whether a frame looks like a parking area.

ponytail: heuristic, no scene classifier. It counts confident slot/vehicle detections, so a street with parked cars
can pass, and with COCO weights (vehicles only) an empty lot is rejected. Add a scene classifier if measurements show it matters.
"""
from __future__ import annotations


def check(detections: list[dict], min_count: int = 2, min_conf: float = 0.4) -> dict:
    """detections from YoloDetector -> {"is_parking_area": bool, "confidence": float, "reason": str}"""
    strong = [d["conf"] for d in detections if d["conf"] >= min_conf]
    if len(strong) < min_count:
        best = max((d["conf"] for d in detections), default=0.0)
        return {"is_parking_area": False, "confidence": round(best, 3),
                "reason": f"only {len(strong)} slot/vehicle detection(s) with confidence >= {min_conf:.2f} "
                          f"(need {min_count}); not a parking area, or the model can't see it"}
    conf = sum(strong) / len(strong)
    return {"is_parking_area": True, "confidence": round(conf, 3),
            "reason": f"{len(strong)} slot/vehicle detections, mean confidence {conf:.2f}"}
