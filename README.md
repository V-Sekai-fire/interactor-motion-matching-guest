# interactor-motion-matching-guest

`mm.elf`: motion matching as a character controller, in a
[godot-sandbox](https://github.com/V-Sekai-fire/godot-sandbox) guest. It is a port of
[godot-motion-matching](https://github.com/V-Sekai/godot-motion-matching) (Guilherme Sousa and
contributors, MIT) without godot-cpp: the pose database, the trajectory and bone features, standard
normalisation and weights, the weighted k-d tree, and the velocity spring that predicts the
trajectory. `src/mm_core.hpp` holds the algorithm; `guest/mm/main.cpp` is only the guest API.

## The motion database

`data/kimodo-locomotion/` holds 36 clips of 10 s at 30 fps on the SOMA skeleton (30 joints), made
with Kimodo (interactor-motion-guest's `motion.elf`, Kimodo-SOMA-RP-v1.1, 50 steps, seed 1171)
from the prompts in `data/kimodo-locomotion/prompts.tsv`: standing, walking at three speeds,
jogging, running, curves, circles, zigzag, walking back, side steps, turns in place, starts,
stops, speed changes and sustained jogs and runs. They are the output of a program; no person recorded them.

`data/o3de-motionmatching/` holds 21 motion-capture clips of the
[O3DE MotionMatching gem](https://github.com/o3de/o3de/tree/development/Gems/MotionMatching/Assets/Animations)
(walks, jogs, runs, pivots, turns on the spot, circles, snake, crouching, jumps; about 24 minutes,
Apache-2.0 OR MIT, see `data/o3de-motionmatching/LICENSE.txt`), moved from its Rin skeleton to
SOMA30 by `tools/fbx_to_soma.gd` (both rest in a T-pose, Y up, facing +Z, so each joint keeps its
world rotation from rest). `Pushes1` is left out: it is the character being pushed.

## Kinds of pose

`mm_bake()` gives every pose a kind from its shape: crouching when the head is below 82% of a
standing head, crawling below 45%, in the air when both feet leave the ground. The host asks for
a kind (`mm_step` mode: 0 stand, 1 crouch, 2 crawl) and only poses of that kind are matched. A jump
(`mm_step` jump = 1) matches a take-off pose (0.1 to 0.45 s before the feet leave) and plays the
jump through to its landing before the next search.

## Use

A host (entities-model-explorer `MotionMatch.gd`) moves the same clips onto its rig, then:

```
mm_add_clip(name, names, parents, transforms, fps)  per clip
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
```

`tests/run_mm.gd` runs the same controller cases in the sandbox; its `TRACE` lines must equal the
native test's.

Kimodo-SOMA-RP-v1.1 is NVIDIA's model under the NVIDIA Open Model License; the clips are its output.
Prompt inputs were made with an LLM2Vec encoder built on Meta Llama 3 (not part of this repository).
