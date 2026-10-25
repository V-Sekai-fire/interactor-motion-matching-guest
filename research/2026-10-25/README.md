# Research notes, 2026-10-25

Raw material behind the walk-bridges and Kimodo speed work, kept so it outlives the scratch dirs.
Nothing here is wired into the pipeline.

## still-all-fours

The three still hands-and-knees bridge ends (crawl_to_stand@0, crouch_to_crawl@299,
stand_to_crawl@299) had no database frame closer than 288 mm mean. Three prompts for a still
all-fours clip were generated with Kimodo-SOMA-RP on ggml-rd (motion.elf, local RenderingDevice,
10 s, seed 1171, 50 steps), plus a `crouch_idle` regeneration. Prompts are in `tries.tsv`; clips in
`clips/<key>/` (transforms.f32, skeleton.txt, parents.txt).

| Clip | Root travel | Root sway max | Hip height | Hands / knees height |
|---|---|---|---|---|
| crawl_idle_a | 0.01 m | 0.007 m | 0.47 m | on the floor |
| crawl_idle_b | 0.01 m | 0.008 m | 0.48 m | on the floor |
| crawl_idle_c | 0.01 m | 0.008 m | 0.47 m | on the floor |

`crawl_idle_a` is the pick. Not yet added to `data/` and the joins not yet re-run.

## kimodo-timing

`GEN-OK` and per-step lines from the speed runs on the M2 Pro (7 bridge prompts, 50 steps):

| Run | 7 clips | Per step | Bytes vs old7 |
|---|---|---|---|
| old7: a sandbox per clip | 1020.6 s | 1.28 s | reference |
| new7b1: one sandbox, batch 1 | 455.7 s | 1.19 s | identical |
| new7b3: batch 3 | 465.8 s | 3.95 s (1.32 per prompt) | |
| new7b7: batch 7 | 510.3 s | 9.13 s (1.30 per prompt) | differ |

GPU compute is about 1.05 s of each step; host copies about 1 ms. Batching gives no gain on this
GPU and changes the bytes at batch 7.

## twist

`twist_cases.py` and `twist_modes.py` compare SOMA-X's aligned-mode twist angle with the standard
swing-twist split on the bridge clips. They are exploratory Python, kept as found.
