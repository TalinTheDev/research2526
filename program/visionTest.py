from vision import FrameSource, FrameProcessor, Visualizer, Detector

detector = Detector(model_path="yolov8n.onnx", confidence_thresh=0.75, device="cpu")
processor = FrameProcessor(detector=detector)
viz = Visualizer(scale=0.6)  # scale down if your webcam is high-res

source = FrameSource(0)
for raw in source:
    fd = processor.process(raw)
    viz.show(raw.image, fd)
    if viz.should_quit():
        break

viz.close()