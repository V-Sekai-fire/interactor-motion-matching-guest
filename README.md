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

## Use

A host (entities-model-explorer `MotionMatch.gd`) moves the same clips onto its rig, then:

```
mm_add_clip(name, names, parents, transforms, fps)  per clip
mm_bake()                                          -> "BAKED clips C poses N dims D"
mm_reset(x, z, yaw, clip)
mm_step(dt, vx, vz, facing)                        per frame -> clip, time, character, clip root, ...
```

## Build and check

```
elixir tools/build.exs --build=<dir>          # mm.elf, with the riscv64-sysroot checkout's clang on PATH
c++ -std=c++20 -O2 -ffp-contract=off -Isrc tests/mm_core_test.cpp -o mm_core_test
./mm_core_test data/kimodo-locomotion          # MM-CORE-PASS
./mm_core_test data/kimodo-locomotion --plant-kd   # must fail: a k-d search that skips the far side
```

`tests/run_mm.gd` runs the same controller cases in the sandbox; its `TRACE` lines must equal the
native test's.

Kimodo-SOMA-RP-v1.1 is NVIDIA's model under the NVIDIA Open Model License; the clips are its output.
Prompt inputs were made with an LLM2Vec encoder built on Meta Llama 3 (not part of this repository).
