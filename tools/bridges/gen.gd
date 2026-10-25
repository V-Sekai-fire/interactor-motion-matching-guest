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


func say(_t: String) -> void:
	pass


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
	var bad := 0
	for line in FileAccess.get_file_as_string(a.prompts).split("\n", false):
		var w := line.split("\t")
		if w.size() == 2 and not await _one(a, w[0], w[1]):
			bad += 1
	print("GEN-DONE bad=%d" % bad)
	get_tree().quit(1 if bad else 0)


func _one(a: Dictionary, key: String, prompt: String) -> bool:
	var t0 := Time.get_ticks_msec()
	if _rd == null:
		_rd = RenderingServer.create_local_rendering_device()
	if _rd == null:
		_fail("no local RenderingDevice")
		return false
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
		return false
	sb.vmcall("motion_env", "KIMODO_STEPS=%d MOTION_GAS=0" % STEPS)
	var av := _avatar()
	sb.vmcall("motion_avatar", av.names, av.parents, av.rest_global, av.rest_local, av.roles)
	var emb := FileAccess.get_file_as_bytes(a.emb.path_join(key + ".f32"))
	if emb.size() != 4096 * 4:
		_fail("no text input for " + key)
		return false
	sb.vmcall("motion_prompt_embedding", prompt, emb.to_float32_array())
	var q := str(sb.vmcall("motion_generate", "kimodo_soma", prompt, _seconds, _seed))
	if not q.begins_with("QUEUED"):
		_fail(q)
		return false
	var id := int(q.split(" ")[1])
	if not await _drive(sb, "motion_pump", key, t0):
		return false
	var st := str(sb.vmcall("motion_status", id))
	if not st.begins_with("DONE"):
		_fail(st)
		return false
	var res: Dictionary = sb.vmcall("motion_result", id)
	remove_child(sb)
	sb.free()
	for f in _files.values():
		f.close()
	_files.clear()
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
				say("%s, %d s" % [label, (Time.get_ticks_msec() - t0) / 1000])
				await get_tree().process_frame
			READ:
				feed = _read(str(r[1]), hdr[1], hdr[2] if hdr[2] > 0 else 1 << 40)
			UPLOAD:
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
