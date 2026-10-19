// mm.elf: motion matching as a character controller, in a godot-sandbox guest.
// The algorithm is src/mm_core.hpp, a port of godot-motion-matching (MIT); this file is only the
// guest API. CPU only, no model weights: the database is a set of SOMA30 clips (Kimodo output in
// motion.elf's motion_result layout), and the host plays the same clips retargeted to its rig.
//
// Host side (entities-model-explorer vsk_model_explorer/core/MotionMatch.gd):
//   mm_add_clip(name, names, parents, transforms, fps)  once per clip -> "ADDED <name> frames N"
//   mm_bake()                                    -> "BAKED clips C poses N dims D"
//   mm_clips()                                   -> one line per clip: name frames seconds speed_m_s
//                                                dx dz (root travel, first to last frame),
//                                                median and 90th-percentile ground speed
//   mm_reset(x, z, yaw, clip)                    start the character there, playing `clip`
//   mm_step(dt, vx, vz, facing)                  one frame: the wanted ground velocity (m/s,
//                                                SOMA metres, world), facing = strafe yaw or
//                                                NaN to face the way it moves -> 13 floats:
//     clip, time, char_x, char_z, char_yaw, root_x, root_z, root_yaw, switched, cost,
//     sim_x, sim_z, sim_yaw
//   The host places the rig so the clip's root at `time` (root_*) lands on the character
//   (char_*): model = T(char) Ry(char_yaw) Ry(-root_yaw) T(-root).
#include <api.hpp>

#include "mm_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static mm::Database g_db;
static mm::Controller g_ctl;
static bool g_baked = false;

static Variant text(const std::string &s) { return Variant(String(s)); }

static Variant mm_clear() {
	g_db = mm::Database();
	g_ctl = mm::Controller();
	g_baked = false;
	return text("CLEARED");
}

static Variant mm_add_clip(String name, String names, PackedArray<int32_t> parents, PackedArray<float> transforms, double fps) {
	std::vector<std::string> nm;
	const std::string all = names.utf8();
	for (size_t i = 0; i < all.size();) {
		const size_t j = std::min(all.find('\n', i), all.size());
		if (j > i) {
			nm.push_back(all.substr(i, j - i));
		}
		i = j + 1;
	}
	g_baked = false;
	return text(g_db.add_clip(name.utf8(), nm, parents.fetch(), transforms.fetch(), float(fps)));
}

static Variant mm_bake() {
	if (g_db.clips.empty()) {
		return text("FAIL no clips");
	}
	const std::string r = g_db.bake();
	g_baked = r.rfind("BAKED", 0) == 0;
	g_ctl = mm::Controller();
	g_ctl.db = &g_db;
	g_ctl.reset({}, 0, 0);
	return text(r);
}

static Variant mm_clips() {
	std::string out;
	char line[256];
	for (const mm::Clip &c : g_db.clips) {
		const float secs = float(c.frames - 1) / c.fps;
		const float speed = secs > 0 ? mm::len(c.root.back() - c.root.front()) / secs : 0.0f;
		const mm::v3 d = c.root.back() - c.root.front();
		std::vector<float> v;
		for (int f = 0; f + 1 < c.frames; ++f) {
			v.push_back(mm::len(c.root[size_t(f) + 1] - c.root[size_t(f)]) * c.fps);
		}
		std::sort(v.begin(), v.end());
		const float med = v.empty() ? 0.0f : v[v.size() / 2];
		const float p90 = v.empty() ? 0.0f : v[std::min(v.size() - 1, v.size() * 9 / 10)];
		std::snprintf(line, sizeof(line), "%s %d %.3f %.3f %.4f %.4f %.3f %.3f\n", c.name.c_str(), c.frames, double(secs),
				double(speed), double(d.x), double(d.z), double(med), double(p90));
		out += line;
	}
	return text(out);
}

static Variant mm_reset(double x, double z, double yaw, int64_t clip) {
	if (!g_baked) {
		return text("FAIL not baked");
	}
	const int c = int(clip) >= 0 && size_t(clip) < g_db.clips.size() ? int(clip) : 0;
	g_ctl.reset(mm::v3{float(x), 0, float(z)}, float(yaw), c);
	return text("RESET " + g_db.clips[size_t(c)].name);
}

static Variant mm_step(double dt, double vx, double vz, double facing) {
	std::vector<float> out;
	if (g_baked) {
		const bool sw = g_ctl.update(float(dt), mm::v3{float(vx), 0, float(vz)}, float(facing));
		const mm::Clip &c = g_db.clips[size_t(g_ctl.clip)];
		const mm::v3 r = mm::Database::sample_root(c, g_ctl.time);
		out = {float(g_ctl.clip), g_ctl.time, g_ctl.char_pos.x, g_ctl.char_pos.z, g_ctl.char_yaw, r.x, r.z,
				mm::Database::sample_yaw(c, g_ctl.time), sw ? 1.0f : 0.0f, g_ctl.last_cost, g_ctl.sim_pos.x,
				g_ctl.sim_pos.z, g_ctl.sim_yaw};
	}
	return Variant(PackedArray<float>(out));
}

int main() {
	ADD_API_FUNCTION(mm_clear, "String", "", "Forget every clip");
	ADD_API_FUNCTION(mm_add_clip, "String", "String name, String names, PackedInt32Array parents, PackedFloat32Array transforms, float fps",
			"A SOMA30 clip (motion.elf motion_result layout) for the pose database");
	ADD_API_FUNCTION(mm_bake, "String", "", "Build the pose database and the k-d tree");
	ADD_API_FUNCTION(mm_clips, "String", "", "name frames seconds speed per clip");
	ADD_API_FUNCTION(mm_reset, "String", "float x, float z, float yaw, int clip", "Place the character, playing a clip");
	ADD_API_FUNCTION(mm_step, "PackedFloat32Array", "float dt, float vx, float vz, float facing",
			"One frame of the controller: clip, time, character, clip root, switched, cost, simulation");
	halt();
}
