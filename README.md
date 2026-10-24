# interactor-motion-matching-guest

`mm.elf`: motion matching as a character controller, in a
[godot-sandbox](https://github.com/V-Sekai-fire/godot-sandbox) guest. It is a port of
[godot-motion-matching](https://github.com/V-Sekai/godot-motion-matching) (Guilherme Sousa and
contributors, MIT) without godot-cpp: the pose database, the trajectory and bone features, standard
normalisation and weights, the weighted k-d tree, and the velocity spring that predicts the
trajectory. `src/mm_core.hpp` holds the algorithm; `guest/mm/main.cpp` is only the guest API.

## The motion database

`data/kimodo-locomotion/` holds 36 clips at 30 fps on the SOMA skeleton (30 joints), each made 10 s long
with Kimodo (interactor-motion-guest's `motion.elf`, Kimodo-SOMA-RP-v1.1, 50 steps, seed 1171)
from the prompts in `data/kimodo-locomotion/prompts.tsv`: standing, walking at three speeds,
jogging, running, curves, circles, zigzag, walking back, side steps, turns in place, starts,
stops, speed changes and sustained jogs and runs. They are the output of a program; no person recorded them.

`data/o3de-motionmatching/` holds 21 motion-capture clips of the
[O3DE MotionMatching gem](https://github.com/o3de/o3de/tree/development/Gems/MotionMatching/Assets/Animations)
(walks, jogs, runs, pivots, turns on the spot, circles, snake, crouching, jumps; about 24 minutes,
Apache-2.0 OR MIT, see `data/o3de-motionmatching/LICENSE.txt`), moved from its Rin skeleton to
SOMA30 by `tools/fbx_to_soma.gd` (both rest in a T-pose, Y up, facing +Z, so each joint keeps its
world rotation from rest). `Pushes1` is left out: it is the character being pushed. The clips
keep the gem's recorded 60 fps, and each recording's opening T-pose frame is dropped.

Each clip stays at the rate it was made at, so the set mixes 30 and 60 fps; nothing is
resampled. The core works in seconds: bone velocities are central differences over 1/30 s
each side read between frames, and the facing averages 0.25 s each side with the edge frames
weighted by the part inside. `mm_core_test`'s RATE check holds each 60 fps clip against its
own every-second frame read as a 30 fps clip (features within 0.25 of a spread, phase within
0.04 of a cycle, 99th percentile); `--plant-rate` reads that twin at the wrong rate and must
fail.

## Steady gaits are loops

`data/kimodo-locomotion/loops.txt` lists the steady gaits. `tools/make_loops.cpp` cuts each
from one left-foot strike to a later one at least two gait cycles on, with every frame between
inside a cycle, choosing the run with the smallest seam; the last 0.2 s fade into the frames
before the cut's start, moved on by one pass's turn and shift. A clip with no such run, or whose
seam (the largest joint turn over the fade and the wrap) is over its ordinary frame step (the
95th percentile of its steps), stays a one-shot; a cut clip gets a `loop.txt`. The host passes the
loop flag to `mm_add_clip`: a loop wraps at its end with the root carried on, and its end forces
no search.

Every frame has a gait phase, 0 to 1 from one left-foot strike to the next (none outside a
cycle). A gait is cyclic, so going back to an earlier stride is an ordinary join, but a switch
between two cycling poses must keep the phase within 0.06 of a cycle (RFD66606-2327, which plans a
learned phase in place of the foot strikes).

## Kinds of pose

`mm_bake()` gives every pose a kind from its shape: crouching when the head is below 82% of a
standing head, crawling below 45%, in the air when both feet leave the ground. The host asks for
a kind (`mm_step` mode: 0 stand, 1 crouch, 2 crawl) and only poses of that kind are matched. A jump
(`mm_step` jump = 1) matches a take-off pose (0.1 to 0.45 s before the feet leave) and plays the
jump through to its landing before the next search.

## Use

A host (entities-model-explorer `MotionMatch.gd`) moves the same clips onto its rig, then:

```
mm_add_clip(name, names, parents, transforms, fps, loop)  per clip
mm_bake()                                          -> "BAKED clips C poses N dims D"
mm_reset(x, z, yaw, clip)
mm_step(dt, vx, vz, facing, mode, jump)            per frame -> clip, time, character, clip root, ...
mm_modes()                                         per kind: poses, usual speed
```

## Build and check

```
elixir tools/build.exs --build=<dir>          # mm.elf, with the riscv64-sysroot checkout's clang on PATH
c++ -std=c++20 -O2 -ffp-contract=off -Isrc tests/mm_core_test.cpp -o mm_core_test
./mm_core_test data/kimodo-locomotion:data/o3de-motionmatching            # MM-CORE-PASS
./mm_core_test data/kimodo-locomotion:data/o3de-motionmatching --plant-kd # must fail
./mm_core_test data/kimodo-locomotion:data/o3de-motionmatching --plant-phase # must fail
c++ -std=c++20 -O2 -Isrc tools/make_loops.cpp -o make_loops
./make_loops data/kimodo-locomotion data/kimodo-locomotion/loops.txt          # cut (in place)
./make_loops data/kimodo-locomotion data/kimodo-locomotion/loops.txt --check  # LOOPS-OK
./make_loops data/kimodo-locomotion data/kimodo-locomotion/loops.txt --uncut  # must fail
```

`tests/run_mm.gd` runs the same controller cases in the sandbox; its `TRACE` lines must equal the
native test's.

Kimodo-SOMA-RP-v1.1 is NVIDIA's model under the NVIDIA Open Model License; the clips are its output.
Prompt inputs were made with an LLM2Vec encoder built on Meta Llama 3 (not part of this repository).
