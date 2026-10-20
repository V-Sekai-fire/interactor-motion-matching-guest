extends SceneTree
# O3DE MotionMatching gem clips (Rin skeleton, FBX) to SOMA30 clips in motion.elf's layout, so they
# join the Kimodo clips in the same database and go onto a rig through the same retarget.
# Both skeletons rest in a T-pose, Y up, facing +Z, left on +X, so each SOMA joint takes its Rin
# joint's world rotation relative to Rin's rest (SOMA's rest rotations are identity), and the hips
# take Rin's pelvis position.
#   godot --headless --path <any project> --script tools/fbx_to_soma.gd ++ \
#     --soma=<a Kimodo clip dir> --out=<dir> [--fps=60] <file.fbx>...
# The gem's clips are recorded at 60 fps, so 60 (the default) samples every recorded frame.

const MAP := {
	"Hips": "C_pelvis_JNT",
	"Spine1": "C_spine_01_JNT",
	"Spine2": "C_spine_02_JNT",
	"Chest": "C_spine_04_JNT",
	"Neck1": "C_neck_01_JNT",
	"Neck2": "C_neck_02_JNT",
	"Head": "C_head_JNT",
	"Jaw": "C_head_JNT",
	"LeftEye": "C_head_JNT",
	"RightEye": "C_head_JNT",
	"LeftShoulder": "L_clavicle_JNT",
	"LeftArm": "L_arm_JNT",
	"LeftForeArm": "L_elbow_JNT",
	"LeftHand": "L_wrist_JNT",
	"LeftHandThumbEnd": "L_wrist_JNT",
	"LeftHandMiddleEnd": "L_wrist_JNT",
	"RightShoulder": "R_clavicle_JNT",
	"RightArm": "R_arm_JNT",
	"RightForeArm": "R_elbow_JNT",
	"RightHand": "R_wrist_JNT",
	"RightHandThumbEnd": "R_wrist_JNT",
	"RightHandMiddleEnd": "R_wrist_JNT",
	"LeftLeg": "L_leg_JNT",
	"LeftShin": "L_knee_JNT",
	"LeftFoot": "L_foot_JNT",
	"LeftToeBase": "L_toe_JNT",
	"RightLeg": "R_leg_JNT",
	"RightShin": "R_knee_JNT",
	"RightFoot": "R_foot_JNT",
	"RightToeBase": "R_toe_JNT",
}


func _arg(n: String) -> String:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--%s=" % n):
			return a.trim_prefix("--%s=" % n)
	return ""


static func _rows(b: Basis) -> PackedFloat32Array:
	# Row-major, acting on column vectors (motion.elf's layout); Basis.x is column 0.
	return PackedFloat32Array([b.x.x, b.y.x, b.z.x, b.x.y, b.y.y, b.z.y, b.x.z, b.y.z, b.z.z])


func _process(_d: float) -> bool:
	var soma_dir := _arg("soma")
	var out_dir := _arg("out")
	var names := PackedStringArray()
	for line in FileAccess.get_file_as_string(soma_dir.path_join("skeleton.txt")).split("\n", false):
		if not line.begins_with("model ") and not line.begins_with("prompt ") and not line.begins_with("frames "):
			names.append(line)
	var parents := PackedInt32Array()
	for p in FileAccess.get_file_as_string(soma_dir.path_join("parents.txt")).split("\n", false):
		parents.append(int(p))
	# SOMA's rest offsets: frame 0's translations of the non-root joints.
	var soma0 := FileAccess.get_file_as_bytes(soma_dir.path_join("transforms.f32")).to_float32_array()
	var offsets: Array[Vector3] = []
	for j in names.size():
		offsets.append(Vector3(soma0[j * 12 + 9], soma0[j * 12 + 10], soma0[j * 12 + 11]))
	var soma_hips_h: float = offsets[0].y
	for path in OS.get_cmdline_user_args():
		if path.begins_with("--"):
			continue
		var doc = ClassDB.instantiate("FBXDocument")
		var st = ClassDB.instantiate("FBXState")
		if doc.append_from_file(path, st) != OK:
			print("SOMA-FAIL cannot read ", path)
			continue
		var scene: Node3D = doc.generate_scene(st)
		root.add_child(scene)
		var sk := scene.find_children("*", "Skeleton3D", true, false)[0] as Skeleton3D
		var ap := scene.find_children("*", "AnimationPlayer", true, false)[0] as AnimationPlayer
		var anim_name: StringName = ap.get_animation_list()[0]
		var anim := ap.get_animation(anim_name)
		var idx := PackedInt32Array()
		var bind_inv: Array[Basis] = []
		for n in names:
			var b := sk.find_bone(MAP[n])
			if b < 0:
				print("SOMA-FAIL %s has no %s" % [path, MAP[n]])
				return _done(1)
			idx.append(b)
			bind_inv.append(sk.get_bone_global_rest(b).basis.orthonormalized().inverse())
		var rin_hips_h := sk.get_bone_global_rest(idx[0]).origin.y
		var scale := soma_hips_h / rin_hips_h
		var FPS := float(_arg("fps")) if _arg("fps") != "" else 60.0
		var frames := int(floor(anim.length * FPS)) + 1
		var xf := PackedFloat32Array()
		xf.resize(frames * names.size() * 12)
		ap.play(anim_name)
		for f in frames:
			ap.seek(min(float(f) / FPS, anim.length), true)
			var g: Array[Basis] = []
			for j in names.size():
				var gb := sk.get_bone_global_pose(idx[j]).basis.orthonormalized()
				g.append(gb * bind_inv[j])
			for j in names.size():
				var local := g[j] if parents[j] < 0 else g[parents[j]].inverse() * g[j]
				var o := (f * names.size() + j) * 12
				var r := _rows(local.orthonormalized())
				for k in 9:
					xf[o + k] = r[k]
				var t := offsets[j]
				if parents[j] < 0:
					t = (sk.global_transform * sk.get_bone_global_pose(idx[0])).origin * scale
				xf[o + 9] = t.x
				xf[o + 10] = t.y
				xf[o + 11] = t.z
		# The recordings open on the T-pose (every joint at rest, hips at the origin); drop those
		# frames so a clip starts on its first recorded pose.
		var first := 0
		while first < frames - 1:
			var rest := true
			for k in names.size() * 12:
				if k % 12 < 9 and absf(xf[first * names.size() * 12 + k] - (1.0 if k % 12 in [0, 4, 8] else 0.0)) > 0.01:
					rest = false
					break
			if not rest:
				break
			first += 1
		if first > 0:
			xf = xf.slice(first * names.size() * 12)
			frames -= first
		var key := path.get_file().get_basename().to_snake_case()
		var dir := out_dir.path_join(key)
		DirAccess.make_dir_recursive_absolute(dir)
		var fx := FileAccess.open(dir.path_join("transforms.f32"), FileAccess.WRITE)
		fx.store_buffer(xf.to_byte_array())
		fx.close()
		var fs := FileAccess.open(dir.path_join("skeleton.txt"), FileAccess.WRITE)
		fs.store_string(
			(
				"model o3de_motionmatching\nprompt %s (O3DE MotionMatching gem, Rin)\nframes %d fps %.1f joints %d\n%s\n"
				% [path.get_file(), frames, FPS, names.size(), "\n".join(names)]
			)
		)
		fs.close()
		var fp := FileAccess.open(dir.path_join("parents.txt"), FileAccess.WRITE)
		for p in parents:
			fp.store_line(str(p))
		fp.close()
		print("SOMA-OK %s frames=%d seconds=%.1f scale=%.3f" % [key, frames, anim.length, scale])
		scene.queue_free()
	return _done(0)


func _done(code: int) -> bool:
	quit(code)
	return true
