# Seeker Robot Algorithm

I wanted to see whether a robot could follow clues instead of just heading toward the first thing its camera recognizes. If you told it to find a kid in red winter clothes, a red hat might be a useful clue, but it isn't the kid. How should the robot tell the difference?

For my 2025–2026 research project, I built a proof-of-concept algorithm tested in a forest simulation to try this idea out. It places a target, clues, and decoys in a seeded scene, with trees that can block the robot's view. The robot decides what to follow using one of four scoring methods: detection confidence, color and label matching, attribute matching, or my clue-aware scorer. I can then compare where each method takes it.

## Try it

You'll need Python 3.12 and a desktop that can show OpenCV windows. From the project folder, create a virtual environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux. Then:

```sh
python -m pip install numpy opencv-python
python program/sim_test.py
```

That runs the same generated scene (seed 44) with all four methods and prints the results. If you want to watch what the robot sees and where it goes, add a seed number:

```sh
python program/sim_test.py 44
```

Change `44` to try a different scene. The visual run pauses after each method; press Space or Q to continue. The output shows whether the robot found the target, how many frames and how much travel it took, and how often it followed the wrong thing. In this simulation, "found" means the target fills at least 4% of the camera view within 2,000 frames. Even the default run creates an OpenCV window, so it needs a graphical desktop session.

## Where the code is

- [`program/sim_test.py`](program/sim_test.py) makes the forest scenes and runs the comparison.
- [`program/custom.py`](program/custom.py) has my clue-aware scoring method; [`program/scoring_interface.py`](program/scoring_interface.py) has the three simpler methods I compare it with.
- [`program/simulation.py`](program/simulation.py) handles movement and results, while [`program/vision/`](program/vision/) handles the simulated camera view and image processing pieces.

## Where I'm at

The comparison currently uses a fixed description of a lost child in red winter clothing. I also experimented with turning a written request into a target description and processing real camera footage, but those pieces aren't connected to the simulation yet. This project shows how the idea behaves in generated scenes, it isn't a test of a physical search robot.

## License

Copyright © 2025–2026 Talin Sharma. All rights reserved. See [`LICENSE`](LICENSE).
