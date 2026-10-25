# A bridge clip's BODY as orbit views (contract-orbit-views STANDARD.md): the SOMA-2603.16858 body
# (SOMA-X, ANNY identity, pose correctives) that body.gd posed on the GPU, drawn as a shaded mesh on a
# ground plane with 1 m squares, one row per frame, one column per camera from
# sphere_hammersley_sequence(i, n, remap=True) around the hips, rendered by SubViewports. Each row is
# labelled with the frame, its time, the motion matcher's kind for it (S stand, C crouch, K crawl,
# A air, B between), the body's top and its lowest vertex (below 0 is into the floor); the cell
# metric is the top. The mesh's vertex buffer takes each frame's posed vertices as they came off the
# GPU (no host maths per vertex); the shading normal is the face normal from screen derivatives.
# Writes the sheet as scene-linear OpenEXR with the lookdev-24 chart, plus .tsv, .chart.tsv, a PNG
# preview, and fails when a view stays blank. --control=zero_views never renders and must fail.
# --movie draws the same views live in the window, one clip frame per rendered frame, for
# Godot's movie writer (--write-movie <stem>.mkv with the godot-cineform engine), and quits after --to.
#   godot --path . --script res://orbit.gd -- --clip=<dir> --body=<body.gd out> --fixture=<dir>
#       --kinds=<file> --out=<stem.exr> --from=<frame> --to=<frame> [--views=4] [--chart=<chart24.json>]
#       [--control=zero_views] [--snap=<png>] [--movie]
extends SceneTree

const THUMB := Vector2i(320, 180)
const LABEL_W := 300
const HEAD_H := 96
const COL_H := 30
const CHART_PATCH := 40
const FOV := 45.0
const LETTERS := {0: "S stand", 1: "C crouch", 2: "K crawl", 4: "A air", 5: "B between"}
const BODY_SHADER := """
shader_type spatial;
render_mode cull_disabled;
uniform vec3 albedo;
void fragment() {
	vec3 n = normalize(cross(dFdx(VERTEX), dFdy(VERTEX)));
	NORMAL = dot(n, VERTEX) > 0.0 ? -n : n;
	ALBEDO = albedo;
	ROUGHNESS = 0.7;
}
"""
# Squares of one metre, so a sliding foot shows against the floor.
const FLOOR_SHADER := """
shader_type spatial;
varying vec3 world;
void vertex() {
	world = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	vec2 c = floor(world.xz);
	ALBEDO = mod(c.x + c.y, 2.0) == 0.0 ? vec3(0.46, 0.46, 0.44) : vec3(0.32, 0.32, 0.31);
	ROUGHNESS = 0.9;
}
"""

var _a := {}
var _frames := 0
var _fps := 30.0
var _kinds := PackedInt32Array()
var _verts := PackedByteArray()
var _box := PackedFloat32Array()
var _hips := PackedFloat32Array()
var _V := 0
var _mesh: ArrayMesh
# The camera distance for the whole clip: the body's largest bounding sphere fills 94% of the frame.
var _reach := 0.0
var _centre_y := 0.0
var _views := []
var _caps := []
var _rows := []
var _frame := 0
var _wait := 0
var _sheet: SubViewport
var _sheet_frames := -1
var _rendered := 0
var _attempted := 0
var _zero := false
var _movie := false
var _movie_started := false
var _missed := 0
var _chart := []
var _chart_at := Vector2i.ZERO
var _status: Label


func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		var kv := arg.trim_prefix("--").split("=", true, 1)
		_a[kv[0]] = kv[1] if kv.size() > 1 else ""
	_zero = _a.get("control", "") == "zero_views"
	_movie = _a.has("movie")
	if not _a.has("clip") or not _a.has("body") or not _a.has("fixture") or not (_a.has("out") or _movie):
		return _fail("usage: --clip=<dir> --body=<dir> --fixture=<dir> --kinds=<file> --out=<stem.exr> --from=F --to=T")
	if not _load_clip(_a.clip):
		return _fail("cannot read the clip or its body " + _a.clip)
	if _a.has("chart"):
		_chart = JSON.parse_string(FileAccess.get_file_as_string(_a.chart)).get("patches", [])
	_views = _hammersley(int(_a.get("views", "4")))
	_build_scene()
	_frame = int(_a.get("from", "0"))
	_pose(_frame)


func _key() -> String:
	return _a.clip.trim_suffix("/").get_file()


func _load_clip(d: String) -> bool:
	var sk := FileAccess.get_file_as_string(d.path_join("skeleton.txt"))
	if sk == "":
		return false
	for line in sk.split("\n", false):
		if line.begins_with("frames "):
			var w := line.split(" ")
			_frames = int(w[1])
			_fps = float(w[3])
	if _a.has("kinds"):
		for k in FileAccess.get_file_as_string(_a.kinds).split("\n", false):
			_kinds.append(int(k))
	_verts = FileAccess.get_file_as_bytes(_a.body.path_join(_key() + ".verts.f32"))
	_box = FileAccess.get_file_as_bytes(_a.body.path_join(_key() + ".bounds.f32")).to_float32_array()
	_hips = FileAccess.get_file_as_bytes(_a.fixture.path_join("clips/%s/transl.bin" % _key())).to_float32_array()
	if _frames == 0 or _box.size() != 6 * _frames or _hips.size() != 3 * _frames or _verts.size() % (12 * _frames) != 0:
		return false
	_V = _verts.size() / (12 * _frames)
	# Framing from the clip's own extent: the largest floor reach from the hips and half the height
	# span over all frames, one distance for the whole clip so rows compare.
	var lo := minf(0.0, _box[1])
	var hi := _box[4]
	for f in _frames:
		var h := Vector2(_hips[3 * f], _hips[3 * f + 2])
		for x in [_box[6 * f], _box[6 * f + 3]]:
			for z in [_box[6 * f + 2], _box[6 * f + 5]]:
				_reach = maxf(_reach, (Vector2(x, z) - h).length())
		lo = minf(lo, _box[6 * f + 1])
		hi = maxf(hi, _box[6 * f + 4])
	_centre_y = (lo + hi) * 0.5
	_reach = maxf(_reach, (hi - lo) * 0.5)
	return true


func _build_scene() -> void:
	var root := Node3D.new()
	get_root().add_child(root)
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.background_color = Color(0.16, 0.17, 0.19)
	env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.environment.ambient_light_color = Color(0.55, 0.55, 0.55)
	root.add_child(env)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, 30, 0)
	sun.shadow_enabled = true
	root.add_child(sun)
	var floor := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	var span := 0.0
	for f in _frames:
		span = maxf(span, maxf(absf(_hips[3 * f]), absf(_hips[3 * f + 2])))
	pm.size = Vector2.ONE * 2.0 * (span + 2.0 * _reach / (0.94 * sin(deg_to_rad(FOV * 0.5))))
	floor.mesh = pm
	var fm := ShaderMaterial.new()
	fm.shader = Shader.new()
	fm.shader.code = FLOOR_SHADER
	floor.material_override = fm
	root.add_child(floor)
	# The body: positions only (stride 12, float32), so each frame's GPU output drops straight into
	# the vertex buffer.
	var faces := FileAccess.get_file_as_bytes(_a.fixture.path_join("faces.i32")).to_int32_array()
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	var pos := PackedVector3Array()
	pos.resize(_V)
	arrays[Mesh.ARRAY_VERTEX] = pos
	arrays[Mesh.ARRAY_INDEX] = faces
	_mesh = ArrayMesh.new()
	_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var bmin := Vector3(INF, INF, INF)
	var bmax := -bmin
	for f in _frames:
		bmin = bmin.min(Vector3(_box[6 * f], _box[6 * f + 1], _box[6 * f + 2]))
		bmax = bmax.max(Vector3(_box[6 * f + 3], _box[6 * f + 4], _box[6 * f + 5]))
	_mesh.custom_aabb = AABB(bmin, bmax - bmin)
	var bm := ShaderMaterial.new()
	bm.shader = Shader.new()
	bm.shader.code = BODY_SHADER
	bm.set_shader_parameter("albedo", Color(0.80, 0.66, 0.58))
	var body := MeshInstance3D.new()
	body.mesh = _mesh
	body.material_override = bm
	root.add_child(body)
	for v in _views:
		var cap := SubViewport.new()
		cap.size = THUMB * 2
		cap.render_target_update_mode = SubViewport.UPDATE_DISABLED if _zero else SubViewport.UPDATE_ALWAYS
		cap.world_3d = get_root().world_3d
		var cam := Camera3D.new()
		cam.fov = FOV
		cap.add_child(cam)
		get_root().add_child(cap)
		_caps.append([cap, cam])
	DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_ALWAYS_ON_TOP, true)
	DisplayServer.window_move_to_foreground()
	# The window shows the views live (a 2x2 grid when there are 4) and what is being drawn.
	var win := DisplayServer.window_get_size()
	var per := int(ceil(sqrt(float(_views.size()))))
	for i in _caps.size():
		var tr := TextureRect.new()
		tr.texture = _caps[i][0].get_texture()
		tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		tr.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
		tr.size = Vector2(win) / per
		tr.position = Vector2((i % per) * win.x / per, (i / per) * win.y / per)
		get_root().add_child(tr)
	var bar := ColorRect.new()
	bar.color = Color(0, 0, 0, 0.75)
	bar.size = Vector2(win.x, 64)
	get_root().add_child(bar)
	_status = Label.new()
	_status.position = Vector2(16, 10)
	_status.add_theme_font_size_override("font_size", 36)
	_status.add_theme_color_override("font_color", Color(1, 1, 1))
	get_root().add_child(_status)


func _pose(f: int) -> void:
	_mesh.surface_update_vertex_region(0, 0, _verts.slice(12 * _V * f, 12 * _V * (f + 1)))
	if _status:
		var k: int = _kinds[f] if f < _kinds.size() else -1
		_status.text = "%s   frame %d of %d   %s   top %.2f m   low %+.3f m" % [_key(), f, _frames - 1,
				LETTERS.get(k, "?"), _box[6 * f + 4], _box[6 * f + 1]]
	var at := Vector3(_hips[3 * f], _centre_y, _hips[3 * f + 2])
	var dist := _reach / (0.94 * sin(deg_to_rad(FOV * 0.5)))
	for i in _views.size():
		var v: Array = _views[i]
		var az := deg_to_rad(v[0])
		var el := deg_to_rad(v[1])
		var dir := Vector3(sin(az) * cos(el), sin(el), cos(az) * cos(el))
		var cam: Camera3D = _caps[i][1]
		var up := Vector3.UP if absf(dir.y) < 0.99 else Vector3.FORWARD
		cam.look_at_from_position(at + dir * dist, at, up)
	_wait = 3


func _process(_dt: float) -> bool:
	# macOS skips drawing an occluded window: then no view renders this iteration, and the movie
	# writer would record the last drawn frame again. Sheets wait for a drawn iteration; a movie
	# counts the misses and fails.
	var drawn_now := DisplayServer.window_can_draw()
	if _movie:
		if not drawn_now:
			_missed += 1
		# The frame posed now is the one this iteration renders and the movie writer records; the
		# first iteration shows --from as _initialize posed it, and the iteration that poses --to
		# quits, so the movie holds frames from..to once each.
		if _movie_started:
			_frame += 1
			_pose(_frame)
		_movie_started = true
		if _frame >= mini(int(_a.get("to", str(_frames - 1))), _frames - 1):
			if _a.has("snap"):
				get_root().get_texture().get_image().save_png(_a.snap)
			print("MOVIE %s frames %d-%d missed %d" % [_key(), int(_a.get("from", "0")), _frame, _missed])
			if _missed > 0:
				return _fail("%d frames were not drawn (window occluded)" % _missed)
			print("RESULT: PASS")
			quit(0)
		return false
	if _sheet != null:
		_sheet_frames += 1
		if _sheet_frames == 3:
			_finish()
		return false
	if not drawn_now:
		return false
	if _wait > 0:
		_wait -= 1
		return false
	var row := {"frame": _frame, "cells": [], "drawn": [], "top": _box[6 * _frame + 4], "low": _box[6 * _frame + 1]}
	for c in _caps:
		_attempted += 1
		var img: Image = c[0].get_texture().get_image()
		var drawn := img != null and not img.is_invisible() and _varied(img)
		if drawn:
			_rendered += 1
		row.cells.append(img if drawn else null)
		row.drawn.append(drawn)
	_rows.append(row)
	if _a.has("snap") and _rows.size() == 2:
		get_root().get_texture().get_image().save_png(_a.snap)
	_frame += 1
	if _frame > int(_a.get("to", str(_frames - 1))) or _frame >= _frames:
		_compose()
	else:
		_pose(_frame)
	return false


# A view counts when it is not one flat colour.
func _varied(img: Image) -> bool:
	var a := img.get_pixel(0, 0)
	for y in range(0, img.get_height(), 2):
		for x in range(0, img.get_width(), 2):
			if not img.get_pixel(x, y).is_equal_approx(a):
				return true
	return false


func _compose() -> void:
	for c in get_root().get_children():
		if c is TextureRect or c is Label or c is ColorRect:
			c.visible = false
	var cols := _views.size()
	var w := LABEL_W + THUMB.x * cols
	var chart_h := CHART_PATCH * 4 + 20 if not _chart.is_empty() else 0
	var h := HEAD_H + COL_H + THUMB.y * _rows.size() + chart_h
	_sheet = SubViewport.new()
	_sheet.size = Vector2i(w, h)
	_sheet.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	get_root().add_child(_sheet)
	_rect(Vector2.ZERO, Vector2(w, h), Color(0.12, 0.12, 0.14))
	var prompt := ""
	for line in FileAccess.get_file_as_string(_a.clip.path_join("skeleton.txt")).split("\n", false):
		if line.begins_with("prompt "):
			prompt = line.substr(7)
	_label("%s   frames %d-%d of %d @ %.0f fps   SOMA-2603.16858 body (ANNY identity, correctives), posed on the GPU" % [
			_key(), _rows[0].frame, _rows[-1].frame, _frames, _fps], Vector2(16, 8), 26)
	_label("%s\n%d of %d views rendered   sphere_hammersley_sequence n=%d   %s" % [prompt, _rendered, _attempted,
			cols, Time.get_datetime_string_from_system(true) + "Z"], Vector2(16, 42), 16)
	for c in cols:
		_label("view %d  az %.1f  el %.1f" % [c, _views[c][0], _views[c][1]], Vector2(LABEL_W + c * THUMB.x + 8, HEAD_H + 4), 16)
	for r in _rows.size():
		var row: Dictionary = _rows[r]
		var y := HEAD_H + COL_H + r * THUMB.y
		var k: int = _kinds[row.frame] if row.frame < _kinds.size() else -1
		_label("frame %d  t %.2f s\n%s\ntop %.2f m\nlow %+.3f m" % [row.frame, row.frame / _fps, LETTERS.get(k, "?"),
				row.top, row.low], Vector2(10, y + 8), 20)
		for c in cols:
			if row.cells[c] == null:
				_label("FAIL: blank", Vector2(LABEL_W + c * THUMB.x + 10, y + 10), 18)
				continue
			var tr := TextureRect.new()
			tr.texture = ImageTexture.create_from_image(row.cells[c])
			tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
			tr.stretch_mode = TextureRect.STRETCH_SCALE
			tr.position = Vector2(LABEL_W + c * THUMB.x, y)
			tr.size = Vector2(THUMB)
			_sheet.add_child(tr)
	if not _chart.is_empty():
		_chart_at = Vector2i(16, HEAD_H + COL_H + THUMB.y * _rows.size() + 10)
		for i in _chart.size():
			var s: Array = _chart[i].srgb8
			_rect(Vector2(_chart_at) + Vector2((i % 6) * CHART_PATCH, (i / 6) * CHART_PATCH), Vector2(CHART_PATCH, CHART_PATCH),
					Color8(int(s[0]), int(s[1]), int(s[2])))
	_sheet_frames = 0


func _rect(at: Vector2, size: Vector2, c: Color) -> void:
	var r := ColorRect.new()
	r.color = c
	r.position = at
	r.size = size
	_sheet.add_child(r)


func _label(text: String, at: Vector2, px: int) -> void:
	var l := Label.new()
	l.text = text
	l.position = at
	l.add_theme_font_size_override("font_size", px)
	l.add_theme_color_override("font_color", Color(0.95, 0.95, 0.95))
	_sheet.add_child(l)


func _finish() -> void:
	var img: Image = _sheet.get_texture().get_image()
	var out: String = _a.out
	DirAccess.make_dir_recursive_absolute(out.get_base_dir())
	img.save_png(out.get_basename() + ".preview.png")
	img.convert(Image.FORMAT_RGBAH)
	img.srgb_to_linear()
	img.save_exr(out)
	var f := FileAccess.open(out.get_basename() + ".tsv", FileAccess.WRITE)
	f.store_line("\t".join(PackedStringArray(["row", "feature", "forecast", "view", "n", "azimuth_deg", "elevation_deg", "metric", "value"])))
	for r in _rows.size():
		for c in _views.size():
			f.store_line("\t".join(PackedStringArray([str(r), "kimodo-bridge-body", "(unforecast)",
					str(c) if _rows[r].drawn[c] else "none", str(_views.size()), "%.6f" % _views[c][0], "%.6f" % _views[c][1],
					"top_m", "%.4f" % _rows[r].top])))
	f.close()
	if not _chart.is_empty():
		var ct := FileAccess.open(out.get_basename() + ".chart.tsv", FileAccess.WRITE)
		ct.store_line("%d %d %d %d" % [_chart_at.x, _chart_at.y, CHART_PATCH * 6, CHART_PATCH * 4])
		ct.close()
	print("ORBIT %s rows %d views %d of %d rendered sheet %dx%d" % [out, _rows.size(), _rendered, _attempted, img.get_width(), img.get_height()])
	if _attempted == 0 or _rendered < _attempted:
		return _fail("%d of %d views blank" % [_attempted - _rendered, _attempted])
	print("RESULT: PASS")
	quit(0)


func _fail(why: String) -> void:
	print("RESULT: FAIL (%s)" % why)
	quit(1)


# sphere_hammersley_sequence(i, n, remap=True) as [azimuth, elevation] in degrees, copied from
# meshing-pen tools/orbit_views.sgd (itself from entities-sakuragaoka-station realize_check.gd).
static func _hammersley(n: int) -> Array:
	var out := []
	for i in n:
		var u := float(i) / n
		var v := 0.0
		var f := 0.5
		var k := i
		while k > 0:
			v += (k & 1) * f
			k >>= 1
			f *= 0.5
		u = 2.0 * u if u < 0.25 else 2.0 / 3.0 * u + 1.0 / 3.0
		var el := rad_to_deg(acos(1.0 - 2.0 * u) - PI / 2.0)
		out.append([v * 360.0, clampf(el, -85.0, 85.0)])
	return out
