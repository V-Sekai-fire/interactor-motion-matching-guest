# Research notes, 2026-11-08

## animation-mixer-migration

Why walk mode's clips should stop being raw files and rejoin Godot's AnimationMixer ecosystem.

### The bug this traces back to

entities-model-explorer#133: `res://motion/mm/clips.tsv` and its `.f32` siblings never landed in
a shipped pck, while `mm.elf` itself always did (confirmed by parsing the released pck for the
real ELF magic bytes, not just the filename string -- an earlier, wrong root-cause guess chased
`ResourceFormatSaverELF` and an `export_presets.cfg` allowlist for a day; both ruled out by a
planted control: populate `motion/mm/` for real and the existing `include_filter` already exports
it correctly). The actual defect is that `tools/package_desktop.exs`'s `copy_mm_data/1` (fed by
`MM_DATA`, built from `interactor-motion-matching-guest`'s `data/` tree) didn't write real clips
into `motion/mm/` for that one CI run -- a script-level failure with no resource-system safety net,
because these are raw files, not Resources.

### Current data flow (the hack)

```
interactor-motion-matching-guest/data/{kimodo-locomotion,o3de-motionmatching}/<clip>/
    transforms.f32, phase.f32, skeleton.txt, parents.txt, loop.txt
        | (tools/release.exs: git archive, mm_data/2)
        v
4-entities/model-explorer res://motion/mm/*.f32, clips.tsv, skeleton.txt, parents.txt
        | (export_presets.cfg include_filter, raw file copy -- not a Resource)
        v
shipped pck: res://motion/mm/*
        | (MotionMatch.gd: FileAccess.file_exists(MM_ELF) and .../clips.tsv)
        v
mm.elf (godot-sandbox guest): reads res://mm.elf's own copy of these files via raw byte I/O,
parses SOMA30 transforms itself (mm_core.hpp), builds its own in-guest KD-tree
```

Every arrow after the first is unverified by Godot's own resource system: nothing but
`contains_all/3`'s substring search (itself fooled once already) checks that these bytes actually
made the trip. `mm_core.hpp`'s own header says why they're raw at all:

> Changed for the guest: clips arrive as SOMA30 transforms (motion.elf's layout) instead of Godot
> Animation tracks

i.e. this was a simplification made specifically for the RISC-V port, not the original design.

### Where it came from

`interactor-motion-matching-guest` is a port of `GuilhermeGSousa/godot-motion-matching` (MIT).
Upstream, `MMAnimationLibrary : AnimationLibrary` bakes its pose database straight from a real
`AnimationMixer`/`Skeleton3D` and real `Animation` resources (`bake_data(character, player,
skeleton)`, reading `get_animation_list()` / `get_animation(name)` off itself as an
`AnimationLibrary`). `MMAnimationNode : AnimationNodeExtension` is the live tree node --
`MotionMatchNode.gd` is already its port. Only the pose database's *source* data got swapped out
for raw files when this went into the sandbox.

### Target data flow

```
interactor-motion-matching-guest/data/.../<clip>/transforms.f32 + friends
        | (build-time GDScript tool, godot --headless --script, matching tools/liltoon_pack.gd's
        |  convention -- NOT Python, per the Lean-4-for-scripting rule)
        v
res://vsk_model_explorer/motion/walk_clips.res  (an AnimationLibrary: one Animation per clip,
    built with add_track + track_insert_key, tracked like any other Resource)
        | (normal Godot export: all_resources picks it up, no include_filter, no copy_mm_data,
        |  no silent-drop risk -- Godot's own resource pipeline is the safety net)
        v
shipped pck: res://vsk_model_explorer/motion/walk_clips.res
        | (mm.elf, via Sandbox's existing Godot-engine-integration syscalls: given a reference to
        |  the AnimationMixer/AnimationLibrary/Skeleton3D, calls get_animation_list/get_animation/
        |  track APIs directly -- no raw file read, no SOMA30 re-parse in guest C++)
        v
mm.elf: bakes/queries the pose database from real Animation data, same as upstream bake_data
```

### Open question: the syscall boundary

Two shapes, not yet decided:

1. **Guest bakes.** Godot hands the guest a `Ref<AnimationLibrary>` (or its Node/Object id) through
   the Sandbox object-call bridge; `mm.elf`'s `bake_data` equivalent calls `get_animation_list` /
   `get_animation(name)->track_get_key_*` through syscalls, same shape as upstream's bake_data, just
   over the sandbox bridge instead of direct C++ calls. Closest port of upstream; keeps the
   KD-tree build inside the guest (today's perf profile, unknown overhead from the syscall traffic
   of pulling every keyframe through the bridge once at bake time, not at 60 Hz).
2. **Godot bakes, guest only queries.** `MotionMatchNode.gd`/a small GDExtension bakes
   `motion_data`/`db_anim_index`/`node_indices` (upstream's own flat arrays, built from the
   AnimationLibrary using Godot's native Animation API, no guest involved) once, then hands those
   flat arrays to `mm.elf` the same way today's `.f32` files do, except as syscall-delivered
   PackedFloat32Array/PackedInt32Array rather than files. `mm.elf`'s runtime `query()` -- the actual
   per-frame KD-tree search -- is unchanged. Smallest guest-side diff; the bake-time-only path
   (once per clip set, not per frame) is the one that would route through Godot's Animation API
   instead of file I/O.

(2) is less guest-code churn and keeps the hot path (per-frame KD-tree search) exactly as-is; it
only moves *where the flat arrays come from*. (1) is a truer port of upstream's architecture, at
the cost of guest-side changes to the bake step.

### What changes, if this proceeds

- `interactor-motion-matching-guest`: `mm_core.hpp`'s raw-file read in the bake step is replaced by
  either (1) or (2) above. The runtime `query()` KD-tree search is untouched either way.
- `entities-model-explorer`: `MotionMatch.gd`'s `MM_ELF`/`DIR`/`available()` checks
  (`res://mm.elf` + `res://motion/mm/clips.tsv`) become `res://mm.elf` + one `AnimationLibrary`
  resource path; `WalkBar.gd`'s gating is unaffected (still keyed off `move.available()`).
- `tools/fetch_guest_assets.sh` / `tools/release.exs`'s `mm_data/2` / `package_desktop.exs`'s
  `copy_mm_data/1`: all three exist only to move raw clip files around; replaced by one build-time
  GDScript conversion step producing the `AnimationLibrary` resource, committed/tracked like any
  other project asset.
- `export_presets.cfg`'s `include_filter` entries for `motion/*.f32`, `motion/mm/*` become
  unnecessary (the resource is exported normally); `*.elf` entries stay (mm.elf itself is still a
  file, just no longer carrying clip data).

### Not yet answered

- Which of the two syscall shapes. Needs someone to actually measure the Sandbox bridge's
  per-call overhead pulling ~72 clips x keyframes once at bake time before picking (1) vs (2).
- Whether `ELFScript`'s own export semantics (checked on this bug hunt, turned out to be a red
  herring for `mm.elf` itself) matter at all once clip data is no longer riding inside the pck as a
  raw file next to it -- probably moot, kept here as a note since it was the first (wrong) theory.
