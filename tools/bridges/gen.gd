extends Node
# The walk-mode bridge clips on the existing Kimodo: motion.elf (interactor-motion-guest, Kimodo on
# ggml-rd) on a local RenderingDevice, as the viewer's Motion.gd runs it: 50 steps, seed 1171, here
# 10 s like the database's clips. The text input of each clip is made beforehand (<emb>/<key>.f32).
#   godot --path . -- --elf=E --weights=W --emb=D --prompts=P --out=O [--seconds=10] [--seed=1171]
# Writes O/<key>/{transforms.f32,skeleton.txt,parents.txt} and prints GEN-OK <key> or GEN-FAIL.
const WAIT_GPU := 1
const READ := 2
const UPLOAD := 3
const COOP := 4
const DONE := 5
const ERROR := 6
const UPLOAD_CHUNK := 32 << 20
const STEPS := 50
var _rd: RenderingDevice
var _files := {}
var _err := ""
var _seconds := 10.0
var _seed := 1171


# The window shows the run: which clip is generating and for how long, and the last finished clip
# playing as a skeleton, so a run is never a blank grey screen.
var _label: Label
var _lines: ImmediateMesh
var _show := []  # per frame, per joint: global position of the last finished clip
var _show_parents := PackedInt32Array()
var _show_t := 0.0
var _a := {}
var _ids := {}  # key -> [guest job id, prompt]
var _saved := {}
var _t0 := 0
var _clip_n := 0
# Arrival time with no prior runs: the guest reports its phase and step i of n; the rate is
# measured live from the phase's first observed step, so remaining = (n - i) / rate. Clips after
# this one take the current clip's projected total, since every clip runs the same model, length
# and step count.
var _phase := ""
var _phase_t0 := 0
var _phase_i0 := 0


func say(t: String) -> void:
	if _label:
		_label.text = t


func _view() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.08, 0.09, 0.11)
	env.ambient_light_color = Color(1, 1, 1)
	env.ambient_light_energy = 0.6
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, 35, 0)
	add_child(sun)
	DisplayServer.window_move_to_foreground()
	var cam := Camera3D.new()
	cam.position = Vector3(2.6, 1.5, 3.2)
	add_child(cam)
	cam.look_at(Vector3(0, 0.7, 0))
	var ground := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(6, 6)
	ground.mesh = pm
	var fm := StandardMaterial3D.new()
	fm.albedo_color = Color(0.22, 0.24, 0.27)
	ground.material_override = fm
	add_child(ground)
	_lines = ImmediateMesh.new()
	var lm := MeshInstance3D.new()
	lm.mesh = _lines
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = Color(0.95, 0.75, 0.35)
	mat.no_depth_test = true
	lm.material_override = mat
	add_child(lm)
	var layer := CanvasLayer.new()
	_label = Label.new()
	_label.position = Vector2(24, 20)
	_label.add_theme_font_size_override("font_size", 26)
	_label.text = "Starting Kimodo"
	layer.add_child(_label)
	add_child(layer)


# Global joint positions of every frame from the clip's local 3x3 + translation rows.
func _globals(tf: PackedFloat32Array, parents: PackedInt32Array, nj: int) -> Array:
	var out := []
	var frames := tf.size() / (nj * 12)
	for fr in frames:
		var g := []
		g.resize(nj)
		for j in nj:
			var o := (fr * nj + j) * 12
			var b := Basis(Vector3(tf[o], tf[o + 3], tf[o + 6]), Vector3(tf[o + 1], tf[o + 4], tf[o + 7]),
					Vector3(tf[o + 2], tf[o + 5], tf[o + 8]))
			var t := Transform3D(b, Vector3(tf[o + 9], tf[o + 10], tf[o + 11]))
			g[j] = t if parents[j] < 0 else (g[parents[j]] as Transform3D) * t
		var pos := PackedVector3Array()
		for j in nj:
			pos.append((g[j] as Transform3D).origin)
		out.append(pos)
	return out


func _process(dt: float) -> void:
	if _lines == null or _show.is_empty():
		return
	_show_t += dt
	var pos: PackedVector3Array = _show[int(_show_t * 30.0) % _show.size()]
	_lines.clear_surfaces()
	_lines.surface_begin(Mesh.PRIMITIVE_LINES)
	for j in pos.size():
		if _show_parents[j] >= 0:
			_lines.surface_add_vertex(pos[_show_parents[j]])
			_lines.surface_add_vertex(pos[j])
	_lines.surface_end()


func _fail(why: String) -> String:
	_err = why
	printerr("GEN-FAIL ", why)
	return ""


func _ready() -> void:
	var a := {}
	for arg in OS.get_cmdline_user_args():
		var kv := arg.trim_prefix("--").split("=", true, 1)
		a[kv[0]] = kv[1] if kv.size() > 1 else ""
	_seconds = float(a.get("seconds", "10"))
	_seed = int(a.get("seed", "1171"))
	_view()
	var jobs := []
	for line in FileAccess.get_file_as_string(a.prompts).split("\n", false):
		var w := line.split("\t")
		if w.size() == 2:
			jobs.append([w[0], w[1]])
	_clip_n = jobs.size()
	var bad := await _all(a, jobs)
	print("GEN-DONE bad=%d" % bad)
	get_tree().quit(1 if bad else 0)


# Every clip in one sandbox: the weights stream to the GPU once, each prompt is queued, and one
# pump drives the guest's queue to the end. A sandbox per clip streamed the weights again for
# every clip.
func _all(a: Dictionary, jobs: Array) -> int:
	var t0 := Time.get_ticks_msec()
	_rd = RenderingServer.create_local_rendering_device()
	if _rd == null:
		_fail("no local RenderingDevice")
		return jobs.size()
	var sb: Node = ClassDB.instantiate("Sandbox")
	sb.memory_max = 2048
	sb.references_max = 65536
	sb.allocations_max = 10000000
	sb.execution_timeout = 0
	add_child(sb)
	sb.load_buffer(FileAccess.get_file_as_bytes(a.elf))
	var opened := str(sb.vmcall("motion_open", _rd, 4096, a.weights))
	if not opened.begins_with("OPEN"):
		_fail(opened)
		return jobs.size()
	sb.vmcall("motion_env", "KIMODO_STEPS=%d MOTION_GAS=0" % STEPS)
	var av := _avatar()
	sb.vmcall("motion_avatar", av.names, av.parents, av.rest_global, av.rest_local, av.roles)
	var ids := {}
	for jb in jobs:
		var emb := FileAccess.get_file_as_bytes(a.emb.path_join(jb[0] + ".f32"))
		if emb.size() != 4096 * 4:
			_fail("no text input for " + jb[0])
			continue
		sb.vmcall("motion_prompt_embedding", jb[1], emb.to_float32_array())
		var q := str(sb.vmcall("motion_generate", "kimodo_soma", jb[1], _seconds, _seed))
		if not q.begins_with("QUEUED"):
			_fail(q)
			continue
		ids[jb[0]] = [int(q.split(" ")[1]), jb[1]]
	_a = a
	_ids = ids
	_t0 = t0
	if not await _drive(sb, "motion_pump", "", t0):
		return jobs.size()
	var bad := jobs.size() - ids.size()
	for key in ids:
		if not _saved.has(key) and not _save(a, sb, key, ids[key][0], ids[key][1], t0):
			bad += 1
	remove_child(sb)
	sb.free()
	_close_files()
	return bad


func _save(a: Dictionary, sb: Node, key: String, id: int, prompt: String, t0: int) -> bool:
	var st := str(sb.vmcall("motion_status", id))
	if not st.begins_with("DONE"):
		_fail(key + ": " + st)
		return false
	_saved[key] = true
	var res: Dictionary = sb.vmcall("motion_result", id)
	var d: String = a.out.path_join(key)
	DirAccess.make_dir_recursive_absolute(d)
	var names: PackedStringArray = str(res.joint_names).strip_edges().split("\n")
	var tf: PackedFloat32Array = res.transforms
	var frames := tf.size() / (names.size() * 12)
	var f := FileAccess.open(d.path_join("transforms.f32"), FileAccess.WRITE)
	f.store_buffer(tf.to_byte_array())
	f.close()
	f = FileAccess.open(d.path_join("skeleton.txt"), FileAccess.WRITE)
	f.store_string("model kimodo_soma\nprompt %s\nframes %d fps %.1f joints %d steps %d seed %d\n" % [prompt, frames, float(res.fps), names.size(), STEPS, _seed])
	for n in names:
		f.store_string(n + "\n")
	f.close()
	f = FileAccess.open(d.path_join("parents.txt"), FileAccess.WRITE)
	for p in res.parents:
		f.store_string(str(p) + "\n")
	f.close()
	_show_parents = PackedInt32Array(res.parents)
	_show = _globals(tf, _show_parents, names.size())
	_show_t = 0.0
	print("GEN-OK %s frames=%d fps=%s joints=%d ms=%d" % [key, frames, str(res.fps), names.size(), Time.get_ticks_msec() - t0])
	return true


# Runs the started job on `pump` to the end, answering its file reads and weight uploads.
func _drive(sb: Node, pump: String, label: String, t0: int) -> bool:
	var feed := PackedByteArray()
	while true:
		var r = sb.vmcall(pump, feed)
		feed = PackedByteArray()
		if typeof(r) != TYPE_ARRAY or r.size() < 3:
			_fail("the program stopped")
			return false
		var hdr: PackedInt64Array = r[0]
		match hdr[0]:
			DONE:
				return true
			ERROR:
				_fail(str(r[1]).strip_edges())
				return false
			WAIT_GPU, COOP:
				say(_status(sb, label, t0))
				await get_tree().process_frame
			READ:
				feed = _read(str(r[1]), hdr[1], hdr[2] if hdr[2] > 0 else 1 << 40)
			UPLOAD:
				say("Kimodo: loading the weights to the GPU (once for all %d clips)" % _clip_n)
				await get_tree().process_frame
				var done := 0
				while done < hdr[2]:
					var n: int = min(hdr[2] - done, UPLOAD_CHUNK)
					var b := _read(str(r[1]), hdr[1] + done, n)
					if b.size() != n or _rd.buffer_update(r[2], hdr[3] + done, n, b) != OK:
						_fail("could not load the weights to the GPU")
						return false
					done += n
			_:
				_fail("unknown request %d" % hdr[0])
				return false
	return false


# Arrival time with no prior runs: the guest reports its job, its phase and step i of n, and how
# many jobs wait. The step rate is measured live within the diffusion phase (the first step seen is
# the start mark, so pipeline set-up before it is not counted); time left = this job's steps left
# plus n steps for each waiting job, divided by that rate. Weight streaming happens once, before the
# first diffusion step, so it is shown but not projected.
func _status(sb: Node, _label: String, t0: int) -> String:
	var now := Time.get_ticks_msec()
	var p = sb.vmcall("motion_progress")
	if typeof(p) != TYPE_DICTIONARY:
		return "Kimodo, %d s" % ((now - t0) / 1000)
	var job := int(p.get("job", -1))
	var ph := str(p.get("phase", ""))
	var i := int(p.get("i", 0))
	var n := int(p.get("n", 0))
	var waiting := int(p.get("queued", 0))
	for key in _ids:  # a finished clip is written and starts playing while the next one runs
		if _ids[key][0] < job and not _saved.has(key):
			_save(_a, sb, key, _ids[key][0], _ids[key][1], _t0)
	var line := "Kimodo: clip %d of %d, %d s\n%s %d of %d" % [job + 1, _clip_n, (now - t0) / 1000, ph, i, n]
	if ph == "diffusion step":
		var done := job * n + i
		if _phase != ph:
			_phase = ph
			_phase_t0 = now
			_phase_i0 = done
		if done > _phase_i0 and now > _phase_t0:
			var rate := float(done - _phase_i0) / float(now - _phase_t0)
			var left_ms := float((n - i) + waiting * n) / rate
			var at := int(Time.get_unix_time_from_system() + left_ms / 1000 + Time.get_time_zone_from_system().bias * 60)
			line += "\n%.2f s a step; about %d s left (arrival %s)" % [1.0 / (rate * 1000), int(left_ms / 1000),
					Time.get_time_string_from_unix_time(at)]
	if not _show.is_empty():
		line += "\nplaying the last finished clip"
	return line


func _read(path: String, offset: int, n: int) -> PackedByteArray:
	if not _files.has(path):
		var f := FileAccess.open(path, FileAccess.READ)
		if f == null:
			return PackedByteArray()
		_files[path] = f
	var file: FileAccess = _files[path]
	file.seek(offset)
	return file.get_buffer(min(n, file.get_length() - offset))


func _close_files() -> void:
	for f in _files.values():
		f.close()
	_files.clear()


# A small T-pose humanoid (motion-guest tests' avatar): motion.elf's generate job needs one.
static func _avatar() -> Dictionary:
	var bones := [
		["Hips", -1, Vector3(0, 1.0, 0), "Hips"],
		["Spine", 0, Vector3(0, 1.1, 0), "Spine"],
		["Chest", 1, Vector3(0, 1.3, 0), "Chest"],
		["Neck", 2, Vector3(0, 1.5, 0), "Neck"],
		["Head", 3, Vector3(0, 1.62, 0), "Head"],
		["L_Arm", 2, Vector3(0.18, 1.45, 0), "LeftUpperArm"],
		["L_Elbow", 5, Vector3(0.46, 1.45, 0), "LeftLowerArm"],
		["L_Hand", 6, Vector3(0.72, 1.45, 0), "LeftHand"],
		["R_Arm", 2, Vector3(-0.18, 1.45, 0), "RightUpperArm"],
		["R_Elbow", 8, Vector3(-0.46, 1.45, 0), "RightLowerArm"],
		["R_Hand", 9, Vector3(-0.72, 1.45, 0), "RightHand"],
		["L_Leg", 0, Vector3(0.1, 0.95, 0), "LeftUpperLeg"],
		["L_Knee", 11, Vector3(0.1, 0.52, 0), "LeftLowerLeg"],
		["L_Foot", 12, Vector3(0.1, 0.08, 0), "LeftFoot"],
		["L_Toes", 13, Vector3(0.1, 0.02, 0.13), "LeftToes"],
		["R_Leg", 0, Vector3(-0.1, 0.95, 0), "RightUpperLeg"],
		["R_Knee", 15, Vector3(-0.1, 0.52, 0), "RightLowerLeg"],
		["R_Foot", 16, Vector3(-0.1, 0.08, 0), "RightFoot"],
		["R_Toes", 17, Vector3(-0.1, 0.02, 0.13), "RightToes"],
	]
	var names := ""
	var roles := ""
	var parents := PackedInt32Array()
	var rg := PackedFloat32Array()
	var rl := PackedFloat32Array()
	for b in bones:
		names += b[0] + "\n"
		roles += b[3] + "\n"
		parents.append(b[1])
		var t: Vector3 = b[2] - (bones[b[1]][2] if b[1] >= 0 else Vector3.ZERO)
		rg.append_array([1, 0, 0, 0, 1, 0, 0, 0, 1, b[2].x, b[2].y, b[2].z])
		rl.append_array([1, 0, 0, 0, 1, 0, 0, 0, 1, t.x, t.y, t.z])
	return {names = names, roles = roles, parents = parents, rest_global = rg, rest_local = rl}
