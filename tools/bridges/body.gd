# The bridge clips posed on the GPU: SOMA-2603.16858 (SOMA-X) with the ANNY identity and its pose
# correctives, through the Lean-emitted kernels of contract-anny-kernels (soma_orient, soma_jump,
# soma_twist, soma_expand, soma_skin, soma_corr_l1, soma_corr_l2, anny_lbs) on a Godot
# RenderingDevice, windowed (never --headless). The host only uploads each frame's pose and reads
# the results; every per-frame body step is a dispatch.
#   godot --path <project> --script res://body.gd -- --fixture=<body_frames.py out> --spv=<spv-anny>
#       --out=<dir> --bridges=<key>,... [--oracle=<key>,...] [--joins]
# Writes <out>/<key>.verts.f32 (F x V x 3, the default Kabsch twist form, correctives on) and
# <key>.bounds.f32 (F x 6: min xyz, max xyz) for every bridge clip; for each --oracle clip also
# <key>.soma.f32 (SOMA-X's half-angle twist form, the form its torch code uses) and <key>.off.f32
# (that form with the correctives off, the parity control); timing.txt (all frames of a clip in one
# submission, pose upload included, best of 5). With --joins: the first and last frame of every bridge
# clip are the join targets, and every fixture clip (the database and the bridges themselves) is
# posed frame by frame and measured against each target as mesh face-vertex distance in the
# character frame (hips position and facing removed): <out>/joins/<key>.f32 (F x targets x 2: max,
# mean in metres). Prints BODY-GPU / JOINS-GPU lines, then RESULT: PASS or RESULT: FAIL (why).
extends SceneTree

const KERNELS := ["soma_orient", "soma_jump", "soma_twist", "soma_expand", "soma_skin", "soma_corr_l1", "soma_corr_l2", "anny_lbs"]
const GROUP := 64
const REPEATS := 5
const CHUNK := 256

const DIST_SRC := """
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std140) uniform P { uint V; uint NTG; uint G; uint pad; } p;
layout(set = 0, binding = 1, std430) readonly buffer Verts { float v[]; };
layout(set = 0, binding = 2, std430) readonly buffer Pose { float pose[]; };
layout(set = 0, binding = 3, std430) readonly buffer Transl { float tr[]; };
layout(set = 0, binding = 4, std430) readonly buffer Tgt { float tv[]; };
layout(set = 0, binding = 5, std430) readonly buffer TgtRoot { float troot[]; };
layout(set = 0, binding = 6, std430) writeonly buffer Part { float part[]; };
shared float smax[64];
shared float ssum[64];
// The character frame: the hips' world x axis (column 0 of its 3x3) projected on the floor gives
// the facing, the hips position on the floor the origin; height stays as it is.
vec3 align(vec3 q, vec3 rx, vec3 h) {
	vec2 r = normalize(vec2(rx.x, rx.z));
	vec2 f = vec2(-r.y, r.x);
	vec2 d = q.xz - h.xz;
	return vec3(dot(d, r), q.y, dot(d, f));
}
void main() {
	uint i = gl_GlobalInvocationID.x;
	uint l = gl_LocalInvocationID.x;
	vec3 crx = vec3(pose[9], pose[12], pose[15]);
	vec3 ch = vec3(tr[0], tr[1], tr[2]);
	for (uint t = 0u; t < p.NTG; ++t) {
		float d = 0.0;
		if (i < p.V) {
			vec3 a = align(vec3(v[3u * i], v[3u * i + 1u], v[3u * i + 2u]), crx, ch);
			uint o = 12u * t;
			vec3 trx = vec3(troot[o], troot[o + 3u], troot[o + 6u]);
			vec3 th = vec3(troot[o + 9u], troot[o + 10u], troot[o + 11u]);
			uint k = 3u * (t * p.V + i);
			vec3 b = align(vec3(tv[k], tv[k + 1u], tv[k + 2u]), trx, th);
			d = distance(a, b);
		}
		smax[l] = d;
		ssum[l] = d;
		barrier();
		for (uint s = 32u; s > 0u; s >>= 1u) {
			if (l < s) {
				smax[l] = max(smax[l], smax[l + s]);
				ssum[l] += ssum[l + s];
			}
			barrier();
		}
		if (l == 0u) {
			uint o = 2u * (t * p.G + gl_WorkGroupID.x);
			part[o] = smax[0];
			part[o + 1u] = ssum[0];
		}
		barrier();
	}
}
"""

const REDUCE_SRC := """
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std140) uniform P { uint V; uint NTG; uint G; uint pad; } p;
layout(set = 0, binding = 1, std430) readonly buffer Part { float part[]; };
layout(set = 0, binding = 2, std430) writeonly buffer Cur { float cur[]; };
void main() {
	uint t = gl_GlobalInvocationID.x;
	if (t >= p.NTG) return;
	float m = 0.0;
	float s = 0.0;
	for (uint g = 0u; g < p.G; ++g) {
		m = max(m, part[2u * (t * p.G + g)]);
		s += part[2u * (t * p.G + g) + 1u];
	}
	cur[2u * t] = m;
	cur[2u * t + 1u] = s / float(p.V);
}
"""

# Per workgroup min/max of x, y, z, then one thread folds the groups: the frame's bounds.
const BOUNDS_SRC := """
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std140) uniform P { uint V; uint G; uint pass_; uint pad; } p;
layout(set = 0, binding = 1, std430) readonly buffer Verts { float v[]; };
layout(set = 0, binding = 2, std430) buffer Part { float part[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Box { float box[]; };
shared vec3 lo[64];
shared vec3 hi[64];
void main() {
	uint i = gl_GlobalInvocationID.x;
	uint l = gl_LocalInvocationID.x;
	if (p.pass_ == 1u) {
		if (i != 0u) return;
		vec3 a = vec3(part[0], part[1], part[2]);
		vec3 b = vec3(part[3], part[4], part[5]);
		for (uint g = 1u; g < p.G; ++g) {
			a = min(a, vec3(part[6u * g], part[6u * g + 1u], part[6u * g + 2u]));
			b = max(b, vec3(part[6u * g + 3u], part[6u * g + 4u], part[6u * g + 5u]));
		}
		box[0] = a.x; box[1] = a.y; box[2] = a.z; box[3] = b.x; box[4] = b.y; box[5] = b.z;
		return;
	}
	uint j = min(i, p.V - 1u);
	vec3 q = vec3(v[3u * j], v[3u * j + 1u], v[3u * j + 2u]);
	lo[l] = q;
	hi[l] = q;
	barrier();
	for (uint s = 32u; s > 0u; s >>= 1u) {
		if (l < s) {
			lo[l] = min(lo[l], lo[l + s]);
			hi[l] = max(hi[l], hi[l + s]);
		}
		barrier();
	}
	if (l == 0u) {
		uint o = 6u * gl_WorkGroupID.x;
		part[o] = lo[0].x; part[o + 1u] = lo[0].y; part[o + 2u] = lo[0].z;
		part[o + 3u] = hi[0].x; part[o + 4u] = hi[0].y; part[o + 5u] = hi[0].z;
	}
}
"""

var rd: RenderingDevice
var _a := {}
var meta: Dictionary
var shaders := {}
var pipes := {}
var fx: String
var out: String
var _status: Label
var _queue := []
var _report := ""
var _fail := ""
# Buffers and sets shared by every step.
var pose: RID
var transl: RID
var verts: RID
var NJ: int
var V: int
var _chain: Callable


func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		var kv := arg.trim_prefix("--").split("=", true, 1)
		_a[kv[0]] = kv[1] if kv.size() > 1 else ""
	fx = _a.get("fixture", "")
	out = _a.get("out", "")
	_status_bar()
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		return _done("no RenderingDevice")
	meta = JSON.parse_string(FileAccess.get_file_as_string(fx.path_join("meta.json")))
	if meta == null:
		return _done("cannot read " + fx.path_join("meta.json"))
	DirAccess.make_dir_recursive_absolute(out.path_join("joins"))
	_setup()
	var bridges: PackedStringArray = _a.get("bridges", "").split(",", false)
	var oracle: PackedStringArray = _a.get("oracle", "").split(",", false)
	for k in bridges:
		_queue.append(_pose_clip.bind(k, k in oracle))
	if _a.has("joins"):
		_queue.append(_join_setup.bind(bridges))
		for c in meta.clips:
			_queue.append(_join_clip.bind(c.key))
	_queue.append(_finish)


func _status_bar() -> void:
	var win := DisplayServer.window_get_size()
	var bar := ColorRect.new()
	bar.color = Color(0.08, 0.08, 0.1)
	bar.size = Vector2(win)
	get_root().add_child(bar)
	_status = Label.new()
	_status.position = Vector2(24, 24)
	_status.add_theme_font_size_override("font_size", 36)
	_status.add_theme_color_override("font_color", Color(1, 1, 1))
	get_root().add_child(_status)


func _process(_dt: float) -> bool:
	if _queue.is_empty():
		return false
	var job: Callable = _queue.pop_front()
	job.call()
	return false


func _done(why: String) -> void:
	print("RESULT: FAIL (%s)" % why if why != "" else "RESULT: PASS")
	quit(1 if why != "" else 0)


func _bytes(path: String) -> PackedByteArray:
	var b := FileAccess.get_file_as_bytes(path)
	if b.is_empty():
		_fail = "cannot read " + path
	return b


func _sbuf(b: PackedByteArray) -> RID:
	var c := b.duplicate()
	c.resize(maxi(16, (b.size() + 15) / 16 * 16))
	return rd.storage_buffer_create(c.size(), c)


func _zeros(n_bytes: int) -> RID:
	var b := PackedByteArray()
	b.resize(n_bytes)
	return _sbuf(b)


func _table(name: String) -> RID:
	return _sbuf(_bytes(fx.path_join("tables/" + name + ".bin")))


func _params(vals: Array) -> RID:
	var b := PackedByteArray()
	b.resize(16)
	for i in vals.size():
		b.encode_u32(4 * i, vals[i])
	return rd.uniform_buffer_create(16, b)


func _shader_spv(k: String) -> void:
	var s := RDShaderSPIRV.new()
	s.set_stage_bytecode(RenderingDevice.SHADER_STAGE_COMPUTE, _bytes(_a.spv.path_join(k + ".spv")))
	_pipe(k, s)


func _shader_glsl(k: String, src: String) -> void:
	var source := RDShaderSource.new()
	source.source_compute = src
	_pipe(k, rd.shader_compile_spirv_from_source(source))


func _pipe(k: String, s: RDShaderSPIRV) -> void:
	if s.compile_error_compute != "":
		_fail = k + ": " + s.compile_error_compute
		push_error(_fail)
		return
	shaders[k] = rd.shader_create_from_spirv(s, k)
	pipes[k] = rd.compute_pipeline_create(shaders[k])


# A uniform set: binding 0 the params block, bindings 1.. the storage buffers in order.
func _uset(k: String, params: RID, bufs: Array) -> RID:
	var us := []
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
	u.binding = 0
	u.add_id(params)
	us.append(u)
	for i in bufs.size():
		var s := RDUniform.new()
		s.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		s.binding = i + 1
		s.add_id(bufs[i])
		us.append(s)
	return rd.uniform_set_create(us, shaders[k], 0)


static func _groups(n: int) -> int:
	return (n + GROUP - 1) / GROUP


func _dispatch(cl: int, k: String, uset: RID, n: int) -> void:
	rd.compute_list_bind_compute_pipeline(cl, pipes[k])
	rd.compute_list_bind_uniform_set(cl, uset, 0)
	rd.compute_list_dispatch(cl, _groups(n), 1, 1)
	rd.compute_list_add_barrier(cl)


func _setup() -> void:
	for k in KERNELS:
		_shader_spv(k)
	_shader_glsl("dist", DIST_SRC)
	_shader_glsl("reduce", REDUCE_SRC)
	_shader_glsl("bounds", BOUNDS_SRC)
	NJ = meta["NJ"]
	V = meta["V"]
	var NT: int = meta["NT"]
	var NS: int = meta["NS"]
	var NTW: int = meta["NTW"]
	var D: int = meta["D"]
	var K: int = meta["K"]
	var rp: int = meta["rounds_pub"]
	var rt: int = meta["rounds_tgt"]
	var orient := _table("orient")
	var orient_pt := _table("orient_pt")
	var pub_local_t := _table("pub_local_t")
	var pub_parent := _table("pub_parent")
	var corr_bind := _table("corr_bind")
	var twist_m := _table("twist_m")
	var val0 := _zeros(4 * 3 * int(meta["E"]))
	pose = _zeros(4 * 9 * NJ)
	transl = _zeros(16)
	verts = _zeros(4 * 3 * V)
	var rabs := _zeros(4 * 9 * NJ)
	var acc := [_zeros(4 * 12 * NJ), _zeros(4 * 12 * NJ)]
	var anc := [_zeros(4 * NJ), _zeros(4 * NJ)]
	var feat := _zeros(4 * 6 * NJ)
	var chan := _zeros(4 * 3 * NJ)
	var tacc := [_zeros(4 * 12 * NT), _zeros(4 * 12 * NT)]
	var tanc := [_zeros(4 * NT), _zeros(4 * NT)]
	var bone := _zeros(4 * 12 * NT)
	var z := _zeros(4 * K)
	var bind := _zeros(4 * 3 * V)
	var pub_world: RID = acc[rp % 2]
	var world: RID = tacc[rt % 2]
	var s_orient := _uset("soma_orient", _params([NJ, int(meta["gtj"])]),
		[pose, transl, orient, orient_pt, pub_local_t, pub_parent, corr_bind, rabs, acc[0], anc[0], feat])
	var p_jp := _params([NJ])
	var s_jp := [_uset("soma_jump", p_jp, [acc[0], anc[0], acc[1], anc[1]]), _uset("soma_jump", p_jp, [acc[1], anc[1], acc[0], anc[0]])]
	var p_jt := _params([NT])
	var s_jt := [_uset("soma_jump", p_jt, [tacc[0], tanc[0], tacc[1], tanc[1]]), _uset("soma_jump", p_jt, [tacc[1], tanc[1], tacc[0], tanc[0]])]
	var tw_bufs := [rabs, pub_world, _table("src_axis"), _table("src_sign"), _table("seg_start"), _table("seg_end"),
		_table("seg_parent"), _table("seg_rev"), _table("bind_rot"), _table("align_rot"), chan]
	var s_twist := [_uset("soma_twist", _params([NJ, NS, 0]), tw_bufs), _uset("soma_twist", _params([NJ, NS, 1]), tw_bufs)]
	var ex_bufs := [_table("tgt_kind"), _table("tgt_ref"), _table("tgt_par"), _table("tgt_base_rot"), _table("tgt_local_t"),
		twist_m, chan, _table("tw_axis"), _table("tw_sign"), pub_world, tacc[0], tanc[0]]
	var s_expand := _uset("soma_expand", _params([NT, NJ, NTW]), ex_bufs)
	var s_skin := _uset("soma_skin", _params([NT]), [world, _table("inv_bind"), bone])
	var s_l1 := _uset("soma_corr_l1", _params([D, K, int(meta["use_tanh"])]), [feat, _table("w1"), z])
	var p_l2 := _params([V])
	var rest := _table("rest")
	var rowptr := _table("w2_rowptr")
	var col := _table("w2_col")
	var s_l2 := [_uset("soma_corr_l2", p_l2, [rest, rowptr, col, _table("w2_val"), z, bind]),
		_uset("soma_corr_l2", p_l2, [rest, rowptr, col, val0, z, bind])]
	var s_lbs := _uset("anny_lbs", _params([V, NT]), [bind, _table("weights"), bone, verts])
	# One frame's chain in the open compute list. form: the twist form (0 Kabsch, the default;
	# 1 SOMA-X's half-angle form); off: correctives off (W2 zero).
	_chain = func(cl: int, form: int, off: bool) -> void:
		_dispatch(cl, "soma_orient", s_orient, NJ)
		for r in rp:
			_dispatch(cl, "soma_jump", s_jp[r % 2], NJ)
		_dispatch(cl, "soma_twist", s_twist[form], NJ)
		_dispatch(cl, "soma_expand", s_expand, NT)
		for r in rt:
			_dispatch(cl, "soma_jump", s_jt[r % 2], NT)
		_dispatch(cl, "soma_skin", s_skin, NT)
		_dispatch(cl, "soma_corr_l1", s_l1, K)
		_dispatch(cl, "soma_corr_l2", s_l2[1 if off else 0], V)
		_dispatch(cl, "anny_lbs", s_lbs, V)


func _clip(key: String) -> Dictionary:
	for c in meta.clips:
		if c.key == key:
			return c
	return {}


func _upload(all_pose: PackedByteArray, all_transl: PackedByteArray, f: int) -> void:
	rd.buffer_update(pose, 0, 4 * 9 * NJ, all_pose.slice(4 * 9 * NJ * f, 4 * 9 * NJ * (f + 1)))
	rd.buffer_update(transl, 0, 12, all_transl.slice(12 * f, 12 * (f + 1)))


var _targets := PackedByteArray()
var _troot := PackedByteArray()
var _tnames := PackedStringArray()


func _pose_clip(key: String, oracle: bool) -> void:
	if _fail != "":
		return
	var c := _clip(key)
	if c.is_empty():
		_fail = key + " is not in the fixture"
		return
	_status.text = "posing %s on the GPU (SOMA-X body, %d frames)" % [key, c.F]
	var F: int = c.F
	var cd := fx.path_join("clips/" + key)
	var all_pose := _bytes(cd.path_join("pose.bin"))
	var all_transl := _bytes(cd.path_join("transl.bin"))
	var G := _groups(V)
	var part := _zeros(4 * 6 * G)
	var box := _zeros(4 * 6)
	var s_b0 := _uset("bounds", _params([V, G, 0]), [verts, part, box])
	var s_b1 := _uset("bounds", _params([V, G, 1]), [verts, part, box])
	var variants := {"verts": [0, false]}
	if oracle:
		variants["soma"] = [1, false]
		variants["off"] = [1, true]
	for name in variants:
		var vv: Array = variants[name]
		var f := FileAccess.open(out.path_join("%s.%s.f32" % [key, name]), FileAccess.WRITE)
		var bf: FileAccess = FileAccess.open(out.path_join(key + ".bounds.f32"), FileAccess.WRITE) if name == "verts" else null
		for fr in F:
			_upload(all_pose, all_transl, fr)
			var cl := rd.compute_list_begin()
			_chain.call(cl, vv[0], vv[1])
			if bf:
				_dispatch(cl, "bounds", s_b0, V)
				_dispatch(cl, "bounds", s_b1, 1)
			rd.compute_list_end()
			rd.submit()
			rd.sync()
			var vb := rd.buffer_get_data(verts, 0, 4 * 3 * V)
			f.store_buffer(vb)
			if bf:
				bf.store_buffer(rd.buffer_get_data(box, 0, 24))
				if fr == 0 or fr == F - 1:
					_targets.append_array(vb)
					_troot.append_array(all_pose.slice(4 * 9 * NJ * fr + 36, 4 * 9 * NJ * fr + 72))
					_troot.append_array(all_transl.slice(12 * fr, 12 * fr + 12))
					_tnames.append("%s@%d" % [key, fr])
		f.close()
		if bf:
			bf.close()
	# Timing: every frame in one submission (pose upload and the whole chain, no readback), best of 5.
	var best := 1e30
	for rep in REPEATS:
		var t0 := Time.get_ticks_usec()
		for fr in F:
			_upload(all_pose, all_transl, fr)
			var cl := rd.compute_list_begin()
			_chain.call(cl, 0, false)
			rd.compute_list_end()
		rd.submit()
		rd.sync()
		best = minf(best, float(Time.get_ticks_usec() - t0) / 1000.0 / F)
	var line := "BODY-GPU %s frames=%d ms_per_frame=%.3f device=%s" % [key, F, best, rd.get_device_name()]
	print(line)
	_report += line + "\n"
	for r in [part, box]:
		rd.free_rid(r)


var _tv: RID
var _tr: RID
var _jpart: RID
var _jcur: RID
var _s_dist: RID
var _s_red: RID
var _NTG := 0


func _join_setup(bridges: PackedStringArray) -> void:
	if _fail != "":
		return
	_NTG = _tnames.size()
	if _NTG != 2 * bridges.size():
		_fail = "join targets: %d, expected %d" % [_NTG, 2 * bridges.size()]
		return
	var G := _groups(V)
	_tv = _sbuf(_targets)
	_tr = _sbuf(_troot)
	_jpart = _zeros(4 * 2 * G * _NTG)
	_jcur = _zeros(4 * 2 * _NTG)
	var p := _params([V, _NTG, G])
	_s_dist = _uset("dist", p, [verts, pose, transl, _tv, _tr, _jpart])
	_s_red = _uset("reduce", p, [_jpart, _jcur])
	var tf := FileAccess.open(out.path_join("joins/targets.txt"), FileAccess.WRITE)
	tf.store_string("\n".join(_tnames) + "\n")
	tf.close()


func _join_clip(key: String) -> void:
	if _fail != "":
		return
	var c := _clip(key)
	var F: int = c.F
	_status.text = "join distances: %s (%d frames) against %d targets" % [key, F, _NTG]
	var cd := fx.path_join("clips/" + key)
	var all_pose := _bytes(cd.path_join("pose.bin"))
	var all_transl := _bytes(cd.path_join("transl.bin"))
	var stride := 4 * 2 * _NTG
	var res := _zeros(stride * F)
	var t0 := Time.get_ticks_usec()
	for start in range(0, F, CHUNK):
		for fr in range(start, mini(start + CHUNK, F)):
			_upload(all_pose, all_transl, fr)
			var cl := rd.compute_list_begin()
			_chain.call(cl, 0, false)
			_dispatch(cl, "dist", _s_dist, V)
			_dispatch(cl, "reduce", _s_red, _NTG)
			rd.compute_list_end()
			rd.buffer_copy(_jcur, res, 0, stride * fr, stride)
		rd.submit()
		rd.sync()
	var f := FileAccess.open(out.path_join("joins/" + key + ".f32"), FileAccess.WRITE)
	f.store_buffer(rd.buffer_get_data(res, 0, stride * F))
	f.close()
	rd.free_rid(res)
	print("JOINS-GPU %s frames=%d ms_per_frame=%.3f" % [key, F, float(Time.get_ticks_usec() - t0) / 1000.0 / F])


func _finish() -> void:
	var tf := FileAccess.open(out.path_join("timing.txt"), FileAccess.WRITE)
	tf.store_string(_report)
	tf.close()
	_done(_fail)
