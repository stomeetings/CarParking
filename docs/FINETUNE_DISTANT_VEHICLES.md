# Detecting distant and angled vehicles: what to change, in the order that pays

Measured 2026-11-02 on `data/sample/Authorities barred from collecting parking fee.png` (640 x 480, an oblique view of a covered row and a road).

## 1. What the weights actually are

* The sample test runs the **profile camera's detector**: stock COCO `yolov8s.pt` at a 1920 px whole-frame pass. It is **not** PKLot-trained.
* The PKLot weights (`models/parking_yolo.pt`, `parking_yolo_obb_*.pt`) are *bay* detectors (`busy_slot` / `free_slot`) learned from top-down lots. On this picture they return **0
  detections** at conf 0.30 and 0.10: the oblique view, the 640 px size and the road / tree clutter are outside what PKLot shows. They cannot fix missed cars; keep them for
  top-down lots (they are listed in the Sample test's model selector, with that note).

## 2. Why distant cars drop out (all measured on that picture)

| Cause | Evidence | Fix |
|---|---|---|
| Confidence 0.30 | vehicles found by yolov8s at 1920 px: 22 at 0.30, 27 at 0.20, 30 at 0.10, 35 at 0.05 | the Sample test now starts at **0.20**; slide down to 0.10 for far rows |
| Parking surface cut | in a calibrated session 26 were detected but 18 counted: the rest stood outside the SAM / drawn surface | the result summary now says "26 found, 8 outside the parking surface or dropped by the filter"; outline more ground |
| Resolution | the picture is 640 px wide; a far car is 20-30 px wide | upscaled 1920 px pass (already default); a bigger model (yolov8x: 25 found at 0.30 vs 22) |
| NMS | boxes of neighbouring cars merge when they overlap > IoU | the IoU slider (0.45 default; 0.55-0.65 for tightly packed rows) |

The detector's confidence and NMS are now per-session settings (`Detection settings` on `#/sample`); nothing live is changed by them.

## 3. If the settings are not enough: fine-tune on your own camera angle

The project already has the pipeline (`scripts/collect_dataset.py` -> `scripts/auto_label.py` -> `yolo detect train`). Add the far-copy step in the middle:

```bash
# 1. frames from the real camera (hours of different light), 2. pre-label with the large teacher and REVIEW the labels
python scripts/collect_dataset.py --camera "Camera 1" --every 60 --hours 48
python scripts/auto_label.py                      # -> datasets/<camera>/{images,labels}/{train,val}
# 3. far-away copies of the train split only (shrunk 0.35-0.65x onto a canvas, labels moved, blurred + re-compressed)
python scripts/augment_far.py datasets/<camera> --copies 2
# 4. train. A larger imgsz keeps far cars at more pixels; start from the stock weights, not from the PKLot ones
yolo detect train data=datasets/<camera>/data.yaml model=yolov8s.pt imgsz=1280 epochs=60 patience=15 batch=8 \
     scale=0.7 mosaic=1.0 close_mosaic=10 degrees=3 hsv_v=0.5 project=experiments/cam name=<camera>
# 5. use it
cp experiments/cam/<camera>/weights/best.pt models/cam_<camera>.pt     # appears in the Sample test's model selector and Mode 5
```

Notes:

* Label **far** cars in the review step. A teacher that misses them teaches the student to miss them; the far copies only help when the shrunk boxes are right.
* Keep the val split real (the script never augments it), or the mAP flatters the model.
* Do not mix PKLot images into a *vehicle* model: PKLot labels bays (`busy_slot` / `free_slot`), not cars. If you want PKLot's variety, convert its occupied-bay boxes to a `car`
  class first and add ~10-20 % of them, never more than your own camera's frames.
* 200-400 reviewed frames of the real view beat thousands of generic ones. Check the new weights on your own held-out frames in the Sample test (pick them in the model selector) and on the val mAP before switching a live camera.
