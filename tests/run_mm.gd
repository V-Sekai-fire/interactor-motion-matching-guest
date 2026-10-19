# Runs mm.elf's controller on the same cases as tests/mm_core_test.cpp and prints the same TRACE
# lines, so the sandbox result can be compared with the native one.
#   godot --headless --path . --script run_mm.gd ++ --elf=<mm.elf> --clips=<dir>
extends SceneTree


func _arg(n: String) -> String:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--%s=" % n):
			return a.trim_prefix("--%s=" % n)
	return ""


func _process(_d: float) -> bool:
	var sb = ClassDB.instantiate("Sandbox")
	sb.set("allocations_max", 1000000)
	sb.set("references_max", 4096)
	sb.set("memory_max", 512)
	sb.set("execution_timeout", 60000)
	sb.load_buffer(FileAccess.get_file_as_bytes(_arg("elf")))
	# --clips may name several dirs, separated by ':' (the native test's order).
	var paths := PackedStringArray()
	for dir in _arg("clips").split(":", false):
		var keys := DirAccess.get_directories_at(dir)
		keys.sort()
		for k in keys:
			paths.append(dir.path_join(k))
	for p_dir in paths:
		var k := p_dir.get_file()
		var sk := FileAccess.get_file_as_string(p_dir.path_join("skeleton.txt")).split("\n", false)
		var fps := 30.0
		var names := PackedStringArray()
		for line in sk:
			if line.begins_with("frames "):
				fps = float(line.split(" ")[3])
			elif not line.begins_with("model ") and not line.begins_with("prompt "):
				names.append(line)
		var parents := PackedInt32Array()
		for p in FileAccess.get_file_as_string(p_dir.path_join("parents.txt")).split("\n", false):
			parents.append(int(p))
		var xf := FileAccess.get_file_as_bytes(p_dir.path_join("transforms.f32")).to_float32_array()
		print(sb.vmcall("mm_add_clip", k, "\n".join(names), parents, xf, fps))
	print(sb.vmcall("mm_bake"))
	# [name, speed, turn_at, turn_to, mode, jump_at] as in tests/mm_core_test.cpp
	var cases := [
		["stand", 0.0, -1.0, 0.0, 0, -1.0],
		["walk", 1.3, -1.0, 0.0, 0, -1.0],
		["run", 3.5, -1.0, 0.0, 0, -1.0],
		["turn", 1.3, 3.0, PI / 2, 0, -1.0],
		["crouch", 0.8, -1.0, 0.0, 1, -1.0],
		["crawl", 0.4, -1.0, 0.0, 2, -1.0],
		["jump", 1.3, -1.0, 0.0, 0, 3.0],
	]
	var modes := str(sb.vmcall("mm_modes"))
	print(modes)
	var dt := 1.0 / 60.0
	var t0 := Time.get_ticks_usec()
	var steps := 0
	for c in cases:
		var have := true
		for line in modes.split("\n", false):
			var w := line.split(" ")
			if (c[4] == 1 and w[0] == "crouch" or c[4] == 2 and w[0] == "crawl" or c[5] >= 0.0 and w[0] == "jump") and int(w[1]) == 0:
				have = false
		if not have:
			continue
		sb.vmcall("mm_reset", 0.0, 0.0, 0.0, 0)
		var jumped := false
		var heading := 0.0
		var out := PackedFloat32Array()
		var sw := 0
		for i in 60 * 8:
			var t := float(i) * dt
			if c[2] >= 0.0 and t >= c[2]:
				heading = c[3]
			var press: int = 1 if c[5] >= 0.0 and not jumped and t >= c[5] else 0
			jumped = jumped or press == 1
			out = sb.vmcall("mm_step", dt, sin(heading) * c[1], cos(heading) * c[1], NAN, c[4], press)
			sw += int(out[8])
			steps += 1
		print("TRACE %s clip=%d time=%.6f x=%.6f z=%.6f yaw=%.6f switches=%d" % [c[0], int(out[0]), out[1], out[2], out[3], out[4], sw])
	print("MM-STEP-US %.1f per step (interpreted)" % (float(Time.get_ticks_usec() - t0) / steps))
	quit(0)
	return true
