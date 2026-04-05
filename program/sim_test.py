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

from vision.sim_world import WorldObject, TreeOccluder, RobotState
from simulation import Simulation
from scoring_interface import ConfidenceOnlyScorer, ColorLabelScorer
from custom import SemanticScorer
from debug_scorer import DebugScorer
import random
import math

DEBUG = False
SEED = 15
TREE_COUNT = 45

TARGET = {
    "object": 'child',
    "state": ['lost'],
    "descriptions": ['red', 'winter clothing', 'young'],
    "associated_objects": ['backpack', 'hat', 'gloves', 'sticks'],
    "colors": ['red'],
    "size_hint": 'small',
    "attributes": ['winter clothing', 'outdoor gear']
}

import random
import math

# ─────────────────────────────────────────────────────────────────────
# LOS CHECK
# ─────────────────────────────────────────────────────────────────────
def is_occluded(p1, p2, trees, buffer=0.2):
    (x1, y1), (x2, y2) = p1, p2

    for tree in trees:
        cx, cy = tree.world_x, tree.world_y
        r = tree.radius + buffer

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            continue

        t = ((cx - x1)*dx + (cy - y1)*dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        if (cx - closest_x)**2 + (cy - closest_y)**2 < r**2:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────
# TREE GENERATION (same logic, slightly compact)
# ─────────────────────────────────────────────────────────────────────
def generate_trees(rng, count=TREE_COUNT):
    trees = []

    def valid(x, y):
        if 6 < x < 9 and 0 < y < 2:
            return False
        for tx, ty in trees:
            if (x - tx)**2 + (y - ty)**2 < 0.8**2:
                return False
        return True

    attempts = 0
    while len(trees) < count and attempts < count * 10:
        x = rng.uniform(0, 22)
        y = rng.uniform(-5, 6)

        if valid(x, y):
            trees.append((x, y))

        attempts += 1

    return [
        TreeOccluder(world_x=x, world_y=y,
                     radius=0.45,
                     height=4.5)
        for x, y in trees
    ]


# ─────────────────────────────────────────────────────────────────────
# CLUE PLACEMENT (visibility-aware)
# ─────────────────────────────────────────────────────────────────────
def place_clue(prev, target, trees, rng):
    for _ in range(50):
        t = rng.uniform(0.3, 0.7)

        x = prev[0] + t * (target[0] - prev[0]) + rng.uniform(-0.8, 0.8)
        y = prev[1] + t * (target[1] - prev[1]) + rng.uniform(-0.8, 0.8)

        if not is_occluded(prev, (x, y), trees):
            return (x, y)

    return (x, y)


def clear_path(p1, p2, trees):
    cleaned = []
    for tree in trees:
        if is_occluded(p1, p2, [tree], buffer=0.3):
            continue
        cleaned.append(tree)
    return cleaned


# ─────────────────────────────────────────────────────────────────────
# MAIN WORLD GENERATOR
# ─────────────────────────────────────────────────────────────────────
def make_world(seed=42):

    rng = random.Random(seed)

    # ── TREES FIRST ──────────────────────────────────────────────────
    trees = generate_trees(rng)

    # ── TARGET ───────────────────────────────────────────────────────
    tx = rng.uniform(21.5, 23.0)
    ty = rng.uniform(4.5, 5.5)
    target_pos = (tx, ty)

    objects = []

    objects.append(
        WorldObject(tx, ty,
                    color="red",
                    shape="rectangle",
                    base_width=0.65,
                    base_height=1.2,
                    label="child",
                    is_target=True)
    )

    # ── CLUE CHAIN (visibility safe) ─────────────────────────────────
    start = (9.5, 1.0)  # just past wall gap
    prev = start

    clue_positions = []
    for _ in range(3):
        pos = place_clue(prev, target_pos, trees, rng)
        clue_positions.append(pos)
        prev = pos

    # clear tree occlusion along path
    for i in range(len(clue_positions) - 1):
        trees = clear_path(clue_positions[i],
                           clue_positions[i+1],
                           trees)

    # ── ADD CLUE OBJECTS ─────────────────────────────────────────────
    labels = ["hat", "jacket", "gloves"]

    for (x, y), label in zip(clue_positions, labels):
        objects.append(
            WorldObject(x, y,
                        color="red",
                        shape="rectangle",
                        base_width=0.5 if label == "hat" else 0.75,
                        base_height=0.35 if label == "hat" else 0.55,
                        label=label)
        )

    # ── DISTRACTORS ──────────────────────────────────────────────────
    colors = ["blue", "green", "yellow", "purple", "orange", "cyan", "red"]
    shapes = ["circle", "triangle", "rectangle"]
    labels = ["vehicle", "animal", "debris", "equipment"]

    for _ in range(12):
        x = rng.uniform(0, 20)
        y = rng.uniform(-5, 6)

        objects.append(
            WorldObject(x, y,
                        color=rng.choice(colors),
                        shape=rng.choice(shapes),
                        base_width=0.7,
                        base_height=0.9,
                        label=rng.choice(labels))
        )

    # hard negatives
    objects.append(
        WorldObject(8, 2,
                    color="blue",
                    shape="rectangle",
                    base_width=0.65,
                    base_height=1.2,
                    label="child")  # false target
    )

    return trees, objects

trees, objects = make_world(seed=SEED)

def main():
    sim = Simulation(
        objects=objects,
        trees=trees,
        camera_width=800,
        camera_height=600,
        fov_degrees=60.0,
        success_fill=0.04,
        max_frames=6000,
        robot_start=RobotState(
            x=0.0, y=0.0,
            heading=0.15,       # nearly east, slight north — will find gap naturally
            speed=0.04,
            turn_rate=0.05,
            search_turn=-0.015,
        ),
        show_window=True,
        window_scale=1.0,
        frame_delay_ms=0,
        pause_on_end=True,
    )

    if DEBUG:
        sim.run(DebugScorer(SemanticScorer(), print_every=30), TARGET)
    else:
        sim.run_comparison(
            algorithms=[
                ConfidenceOnlyScorer(),
                ColorLabelScorer(),
                SemanticScorer(),
            ],
            target=TARGET,
        )


if __name__ == "__main__":
    main()
