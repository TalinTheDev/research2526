"""
sim_test.py
-----------
Prompt: "10 year old boy in red winter clothing lost in the dense forest"

World design — occlusion-first layout:
    The target is hidden behind a dense tree wall until the robot is
    already close. The robot must navigate around the wall, which puts
    it on a path past the trail clues. This is the only layout where
    trail-following behavior is actually required to succeed.

Layout (top-down, x = east, y = north):

    Start(0,0) facing east (heading ~0)

    ZONE 1 (x=0-6): Open area with distractors north and south.
                    Robot searches, may find backpack clue at (4, 0).

    ZONE 2 (x=6-9): Dense tree wall, BLOCKING direct view of target.
                    Robot must navigate around — either north or south gap.

    ZONE 3 (x=9-14): Corridor past the wall. Red hat at (10, 1) and
                     red jacket at (13, 0.5) are here — encountered
                     naturally when navigating through the gap.

    ZONE 4 (x=14-22): Final approach. Target at (20, 0) behind a
                      light tree scatter. Visible once robot is ~x=14.

    DISTRACTORS: Same-label wrong-color children are north (y>4) and
                 south (y<-4) of the main corridor, not in it.
                 Same-color wrong-label objects are scattered throughout
                 but BEHIND the tree wall (x>9) so they don't short-
                 circuit the trail.

Why this tests the algorithm properly:
    - ConfidenceOnly: sees distractors in zone 1, gets trapped there
      or navigates randomly through the wall gap, missing the trail.
    - ColorLabel: may follow the red hat/jacket but treats them as
      targets and stops, never reaching the actual child.
    - SemanticScorer: treats hat/jacket as clues (band 0.45-0.64),
      keeps moving, locks onto red child (band 0.65+) when visible.
"""

import sys

from vision.sim_world import WorldObject, TreeOccluder, RobotState
from simulation import Simulation
from scoring_interface import ConfidenceOnlyScorer, ColorLabelScorer, AttributeCountScorer
from custom import SemanticScorer
from debug_scorer import DebugScorer
import random
import math

DEBUG = False
SEED = 0
TREE_COUNT = 50
SHOW_WINDOW = False
START = 44
COUNT = 1

TARGET = {
    "object": 'child',
    "state": ['lost'],
    "descriptions": ['red', 'winter clothing', 'young'],
    "associated_objects": ['backpack', 'hat', 'gloves', 'sticks'],
    "colors": ['red'],
    "size_hint": 'small',
    "attributes": ['winter clothing', 'outdoor gear']
}

# ─────────────────────────────────────────────────────────────
# CORE IDEA:
# generate a "true path" + "decoy path" in latent space,
# then place objects around those paths with noise
# ─────────────────────────────────────────────────────────────

def make_world(seed=42):

    rng = random.Random(seed)

    trees = []
    objects = []

    # ── STEP 1: Generate forest (background only) ─────────────
    def generate_forest(n=TREE_COUNT):
        pts = []
        for _ in range(n):
            x = rng.uniform(0, 24)
            y = rng.uniform(-6, 6)
            pts.append((x, y))
        return pts

    tree_pts = generate_forest()

    trees = [
        TreeOccluder(x, y, radius=0.45, height=4.5)
        for x, y in tree_pts
    ]

    # ── STEP 2: Define TRUE PATH (latent spline-like path) ─────
    start = (2, rng.uniform(-1, 1))
    target = (22, rng.uniform(2, 4))

    def interpolate_path(p0, p1, steps=4, noise=1.0):
        pts = []
        for i in range(1, steps + 1):
            t = i / (steps + 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])

            # add curvature
            y += math.sin(t * math.pi) * rng.uniform(-noise, noise)

            pts.append((x, y))
        return pts

    true_path = [start] + interpolate_path(start, target, 3) + [target]

    # ── STEP 3: Create CLUES along true path ───────────────────
    clue_labels = ["hat", "jacket", "gloves"]

    for (x, y), label in zip(true_path[1:-1], clue_labels):
        objects.append(
            WorldObject(
                x + rng.uniform(-0.5, 0.5),
                y + rng.uniform(-0.5, 0.5),
                color="red",
                shape="rectangle",
                base_width=0.6,
                base_height=0.5,
                label=label
            )
        )

    # ── STEP 4: TRUE TARGET ───────────────────────────────────
    tx, ty = target
    objects.append(
        WorldObject(
            tx, ty,
            color="red",
            shape="rectangle",
            base_width=0.65,
            base_height=1.2,
            label="child",
            is_target=True
        )
    )

    # ── STEP 5: DECOY PATH (parallel but diverging) ────────────
    offset = rng.choice([-3, 3])
    decoy_target = (target[0] - rng.uniform(3, 5), target[1] + offset)

    decoy_path = [start] + interpolate_path(start, decoy_target, 3)

    for (x, y) in decoy_path[1:]:
        objects.append(
            WorldObject(
                x + rng.uniform(-0.5, 0.5),
                y + rng.uniform(-0.5, 0.5),
                color="red",  # SAME COLOR
                shape="rectangle",
                base_width=0.6,
                base_height=0.5,
                label=rng.choice(["jacket", "coat"])
            )
        )

    # ── STEP 6: FALSE TARGET ──────────────────────────────────
    fx, fy = decoy_target
    objects.append(
        WorldObject(
            fx, fy,
            color="blue",
            shape="rectangle",
            base_width=0.65,
            base_height=1.2,
            label="child",   # identical!
            is_target=False
        )
    )

    # ── STEP 7: RANDOM DISTRACTORS ─────────────────────────────
    colors = ["blue", "green", "yellow", "purple", "orange", "cyan"]
    shapes = ["circle", "triangle", "rectangle"]
    labels = ["vehicle", "animal", "debris", "equipment"]

    for _ in range(15):
        objects.append(
            WorldObject(
                rng.uniform(0, 24),
                rng.uniform(-6, 6),
                color=rng.choice(colors),
                shape=rng.choice(shapes),
                base_width=rng.uniform(0.5, 1.0),
                base_height=rng.uniform(0.5, 1.2),
                label=rng.choice(labels)
            )
        )

    return trees, objects



def main(seed, window, debug):
    trees, objects = make_world(seed=seed)
    sim = Simulation(
        objects=objects,
        trees=trees,
        camera_width=800,
        camera_height=600,
        fov_degrees=90.0,
        success_fill=0.04,
        max_frames=2000,
        robot_start=RobotState(
            x=0.0, y=0.0,
            heading=0.15,       # nearly east, slight north — will find gap naturally
            speed=0.04,
            turn_rate=0.05,
            search_turn=-0.015,
        ),
        show_window=window,
        window_scale=1.0,
        frame_delay_ms=0,
        pause_on_end=window,
    )
    if debug:
        sim.run(DebugScorer(SemanticScorer(), print_every=30), TARGET, debug=debug)
    else:
        sim.run_comparison(
            algorithms=[
                ConfidenceOnlyScorer(),
                ColorLabelScorer(),
                AttributeCountScorer(),
                SemanticScorer(),
            ],
            seed=seed,
            target=TARGET,
            debug=debug,
        )

def run():
    print(f"\n{'='*70}\n  COMPARISON RESULTS\n{'='*70}")
    print(f"  {'Seed,':<10} {'Algorithm,':<20} {'Result,':<10} {'Frames,':>8} {'Path,':>8} {'FalseFollows,':>14} {'FillRatio':>10}")
    
    if len(sys.argv) > 1:
    #    print(f"using seed {sys.argv[1]} and showing window")
       main(int(sys.argv[1]), True, True)
    else:
        for n in range(START, START + COUNT):
            main(n, False, False)

    print(f"{'='*70}\n")

if __name__ == "__main__":
    run()