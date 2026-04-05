"""
vision_test.py
--------------
Ready-to-run test script for the vision module using synthetic frames.

Scene:
    red rectangle   → person   (stationary)
    blue circle     → animal   (moving)
    yellow triangle → debris   (stationary)
    orange rectangle→ landmark (slow drift)
    purple circle   → vehicle  (stationary, large)

Run from your project root:
    python vision_test.py
"""

from vision import FrameProcessor, Visualizer
from vision.color_shape_detector import ColorShapeDetector
from vision.test_frame_generator import TestFrameGenerator, ShapeSpec


def main():
    # ── Scene setup ────────────────────────────────────────────────────
    gen = (
        TestFrameGenerator(width=800, height=600, fps=30.0)
        .add_shape(ShapeSpec(
            color="red", shape="rectangle",
            position=(200, 200), size=(70, 120),
            velocity=(0, 0), label="person",
        ))
        .add_shape(ShapeSpec(
            color="blue", shape="circle",
            position=(450, 180), size=(55, 55),
            velocity=(2.5, 1.2), label="animal",
        ))
        .add_shape(ShapeSpec(
            color="yellow", shape="triangle",
            position=(580, 350), size=(65, 65),
            velocity=(0, 0), label="debris",
        ))
        .add_shape(ShapeSpec(
            color="orange", shape="rectangle",
            position=(150, 420), size=(50, 50),
            velocity=(0.8, 0.4), label="landmark",
        ))
        .add_shape(ShapeSpec(
            color="purple", shape="circle",
            position=(650, 420), size=(90, 90),
            velocity=(0, 0), label="vehicle",
        ))
    )

    # ── Pipeline setup ─────────────────────────────────────────────────
    detector  = ColorShapeDetector(min_area=400, confidence_thresh=0.40)
    processor = FrameProcessor(detector=detector)
    viz       = Visualizer(window_name="Seeker Vision Test", scale=1.0)

    print("Running vision test — press Q or Escape to quit.\n")

    # ── Main loop ──────────────────────────────────────────────────────
    for raw in gen.stream():   # infinite by default, Q to quit
        fd = processor.process(raw)

        # Print detections to console every 30 frames
        if fd.frame_id % 30 == 0:
            print(f"[Frame {fd.frame_id:04d}]  {len(fd.detections)} detection(s)")
            for det in fd.detections:
                print(f"  {det.label:<12} conf={det.confidence:.2f}  "
                      f"attrs={det.attributes}  state={det.state}")
            print()

        viz.show(raw.image, fd)
        if viz.should_quit():
            break

    viz.close()
    print("Done.")


if __name__ == "__main__":
    main()
