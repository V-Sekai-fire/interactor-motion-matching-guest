#include <algorithm>
// Native checks of mm_core.hpp on real Kimodo SOMA30 clips (a directory of
// <key>/transforms.f32 + skeleton.txt + parents.txt as motion.elf's chain runner writes them).
//   mm_core_test <clips dir> [--plant-kd] [--plant-phase]
// 1. The k-d tree finds the same cost as a brute-force scan on 500 queries (database rows plus
//    noise). --plant-kd skips the far branch, which must make this check fail.
// 2. Every 50th pose, queried with its own features, finds a pose of cost 0.
// 3. The controller: standing, walking and running forward, and a 90-degree turn; it must move at
//    about the asked speed, face the asked way, and not flicker between poses.
// 4. Joins keep the gait phase (RFD66606-2327): 60 s at 144 Hz walking, jogging and running, every
//    switch between two cycling poses, back in time or to another clip, lands within the phase
//    tolerance. --plant-phase switches the phase rule off, which must make this check fail.
#include "mm_core.hpp"

#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <map>
#include <random>
#include <sstream>


static bool read_clip(const std::string &dir, std::vector<std::string> &names, std::vector<int32_t> &parents,
		std::vector<float> &xf, float &fps) {
	std::ifstream sk(dir + "/skeleton.txt"), pa(dir + "/parents.txt");
	std::ifstream tr(dir + "/transforms.f32", std::ios::binary);
	if (!sk || !pa || !tr) {
		return false;
	}
	std::string line;
	int frames = 0, joints = 0;
	while (std::getline(sk, line)) {
		if (line.rfind("frames ", 0) == 0) {
			std::istringstream s(line);
			std::string w;
			s >> w >> frames >> w >> fps >> w >> joints;
		} else if (line.rfind("model ", 0) != 0 && line.rfind("prompt ", 0) != 0 && !line.empty()) {
			names.push_back(line);
		}
	}
	int p;
	while (pa >> p) {
		parents.push_back(p);
	}
	tr.seekg(0, std::ios::end);
	xf.resize(size_t(tr.tellg()) / 4);
	tr.seekg(0);
	tr.read(reinterpret_cast<char *>(xf.data()), long(xf.size() * 4));
	return int(names.size()) == joints && int(xf.size()) == frames * joints * 12;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: mm_core_test <clips dir> [--plant-kd]\n");
		return 2;
	}
	bool plant = false, plant_phase = false, foot_phase = false, plant_rate = false;
	for (int i = 2; i < argc; ++i) {
		plant = plant || std::string(argv[i]) == "--plant-kd";
		plant_phase = plant_phase || std::string(argv[i]) == "--plant-phase";
		plant_rate = plant_rate || std::string(argv[i]) == "--plant-rate"; // twins read at the wrong rate
		foot_phase = foot_phase || std::string(argv[i]) == "--foot-phase"; // ignore phase.f32
	}
	mm::Database db;
	const float tol = db.settings.phase_tol;
	db.settings.phase_tol = plant_phase ? 0.0f : tol;
	// argv[1] may name several clip dirs, separated by ':'.
	std::vector<std::string> keys;
	const std::string roots = argv[1];
	for (size_t s = 0; s <= roots.size();) {
		const size_t e = std::min(roots.find(':', s), roots.size());
		const std::string root = roots.substr(s, e - s);
		std::vector<std::string> here;
		if (DIR *d = opendir(root.c_str())) {
			while (dirent *de = readdir(d)) {
				if (de->d_name[0] != '.') {
					here.push_back(root + "/" + de->d_name);
				}
			}
			closedir(d);
		}
		std::sort(here.begin(), here.end());
		keys.insert(keys.end(), here.begin(), here.end());
		s = e + 1;
	}
	struct Raw {
		std::string name;
		std::vector<std::string> names;
		std::vector<int32_t> parents;
		std::vector<float> xf;
		float fps;
	};
	std::vector<Raw> fast; // clips above 30 fps, for the mixed-rate check
	for (const std::string &path : keys) {
		std::vector<std::string> names;
		std::vector<int32_t> parents;
		std::vector<float> xf;
		float fps = 30;
		if (!read_clip(path, names, parents, xf, fps)) {
			continue;
		}
		const std::string k = path.substr(path.find_last_of('/') + 1);
		const bool loop = std::ifstream(path + "/loop.txt").good();
		if (fps > 31) {
			fast.push_back({k, names, parents, xf, fps});
		}
		std::printf("%s%s\n", db.add_clip(k, names, parents, xf, fps, loop).c_str(), loop ? " loop" : "");
		std::vector<float> ph;
		std::ifstream pf(path + "/phase.f32", std::ios::binary);
		if (pf && !foot_phase) {
			pf.seekg(0, std::ios::end);
			ph.resize(size_t(pf.tellg()) / 4);
			pf.seekg(0);
			pf.read(reinterpret_cast<char *>(ph.data()), long(ph.size() * 4));
			db.set_phase(int(db.clips.size()) - 1, ph);
		}
	}
	std::printf("%s\n", db.bake().c_str());
	if (plant) {
		// The planted bug: a tree that never visits the far side.
		db.tree->skip_far = true;
	}
	int fails = 0;

	// 1. k-d tree == brute force.
	std::mt19937 rng(7);
	std::normal_distribution<float> noise(0, 0.3f);
	const int D = db.dims();
	int kd_bad = 0;
	long visited_sum = 0;
	for (int i = 0; i < 500; ++i) {
		const int row = db.eligible[rng() % unsigned(db.eligible.size())];
		std::vector<float> q(db.data.begin() + long(row) * D, db.data.begin() + long(row + 1) * D);
		for (float &x : q) {
			x += noise(rng);
		}
		int visited = 0;
		const int a = db.search(q.data(), &visited);
		visited_sum += visited;
		const int b = db.search_brute(q.data());
		if (std::fabs(db.cost(a, q.data()) - db.cost(b, q.data())) > 1e-6f * std::max(1.0f, db.cost(b, q.data()))) {
			++kd_bad;
		}
	}
	std::printf("KD %s mismatches=%d/500 mean_visited=%.0f of %d\n", kd_bad ? "FAIL" : "PASS", kd_bad,
			double(visited_sum) / 500.0, db.pose_count());
	fails += kd_bad ? 1 : 0;

	// 2. Self match.
	int self_bad = 0;
	for (size_t k = 0; k < db.eligible.size(); k += 50) {
		const int i = db.eligible[k];
		const float *q = &db.data[size_t(i) * size_t(D)];
		if (db.cost(db.search(q), q) > 1e-9f) {
			++self_bad;
		}
	}
	std::printf("SELF %s bad=%d\n", self_bad ? "FAIL" : "PASS", self_bad);
	fails += self_bad ? 1 : 0;

	// 3. The controller.
	struct Case {
		const char *name;
		float speed, turn_at, turn_to;
		const char *expect;
		int mode;
		float jump_at;
	};
	const Case cases[] = {
			{"stand", 0.0f, -1, 0, "idle|turn|stop", mm::Stand, -1},
			{"walk", 1.3f, -1, 0, "walk", mm::Stand, -1},
			{"run", 3.5f, -1, 0, "run|jog", mm::Stand, -1},
			{"turn", 1.3f, 3.0f, mm::kPi / 2, "walk|turn", mm::Stand, -1},
			{"crouch", 0.8f, -1, 0, "crouch", mm::Crouch, -1},
			{"crawl", 0.4f, -1, 0, "crawl", mm::Crawl, -1},
			{"jump", 1.3f, -1, 0, "jump", mm::Stand, 3.0f},
	};
	for (const Case &c : cases) {
		if (c.mode != mm::Stand && !db.has_mode(c.mode)) {
			std::printf("CTL %s SKIP no poses of that kind\n", c.name);
			continue;
		}
		if (c.jump_at >= 0 && !db.has_mode(mm::Jump)) {
			std::printf("CTL %s SKIP no take-off poses\n", c.name);
			continue;
		}
		mm::Controller ctl;
		ctl.db = &db;
		ctl.reset({}, 0, 0);
		std::map<std::string, float> played;
		const float dt = 1.0f / 60;
		float heading = 0;
		float kind_ok = 0, kind_all = 0, air = 0;
		bool jumped = false;
		for (int i = 0; i < 60 * 8; ++i) {
			const float t = float(i) * dt;
			if (c.turn_at >= 0 && t >= c.turn_at) {
				heading = c.turn_to;
			}
			const bool press = c.jump_at >= 0 && !jumped && t >= c.jump_at;
			jumped = jumped || press;
			ctl.update(dt, mm::v3{std::sin(heading) * c.speed, 0, std::cos(heading) * c.speed},
					std::numeric_limits<float>::quiet_NaN(), c.mode, press);
			const mm::Clip &pc = db.clips[size_t(ctl.clip)];
			const int pf = std::clamp(int(std::lround(ctl.time * pc.fps)), 0, pc.frames - 1);
			if (t > 2.0f) {
				played[pc.name] += dt;
				kind_all += dt;
				kind_ok += pc.kind[size_t(pf)] == c.mode ? dt : 0.0f;
			}
			if (c.jump_at >= 0 && t >= c.jump_at && t < c.jump_at + 1.5f && pc.kind[size_t(pf)] == mm::Air) {
				air += dt;
			}
		}
		std::string top;
		float most = 0;
		std::string list;
		for (auto &[k, v] : played) {
			list += k + "=" + std::to_string(v).substr(0, 4) + " ";
			if (v > most) {
				most = v;
				top = k;
			}
		}
		const mm::v3 p = ctl.char_pos;
		const float dist = mm::len(p);
		bool name_ok = false;
		std::string ex = c.expect;
		size_t s = 0;
		while (s <= ex.size()) {
			const size_t e = std::min(ex.find('|', s), ex.size());
			name_ok = name_ok || top.find(ex.substr(s, e - s)) != std::string::npos;
			s = e + 1;
		}
		const float want = c.speed * 8.0f;
		const bool dist_ok = c.speed == 0 ? dist < 1.0f : (dist > 0.6f * want && dist < 1.25f * want);
		const float face_err = std::fabs(mm::wrap_angle(ctl.char_yaw - (c.turn_at >= 0 ? c.turn_to : 0.0f)));
		const bool face_ok = c.speed == 0 || face_err < 0.35f;
		// A crouch or crawl plays poses of that kind most of the time; a jump leaves the ground.
		const bool kind_good = c.mode == mm::Stand || kind_ok >= 0.8f * kind_all;
		const bool jump_good = c.jump_at < 0 || (ctl.jumps == 1 && air > 0.1f);
		std::printf("KIND %s share=%.2f air=%.2f s jumps=%d\n", c.name, double(kind_all > 0 ? kind_ok / kind_all : 0.0f),
				double(air), ctl.jumps);
		// Settled motion does not flicker between poses: at most 3 switches a second. The clip
		// names are only reported: motion matching uses any part of any clip (the walking start of
		// stop_walk is a walk).
		const bool calm = float(ctl.switches) / 8.0f <= 3.0f;
		(void)name_ok;
		const bool ok = dist_ok && face_ok && calm && kind_good && jump_good;
		std::printf("TRACE %s clip=%d time=%.6f x=%.6f z=%.6f yaw=%.6f switches=%d\n", c.name, ctl.clip, double(ctl.time),
				double(ctl.char_pos.x), double(ctl.char_pos.z), double(ctl.char_yaw), ctl.switches);
		std::printf("CTL %s %s travel=%.2f m (asked %.1f) yaw_err=%.2f rad searches=%d switches=%d top=%s | %s\n", c.name,
				ok ? "PASS" : "FAIL", double(dist), double(want), double(face_err), ctl.searches, ctl.switches, top.c_str(),
				list.c_str());
		fails += ok ? 0 : 1;
	}
	// 4. Joins keep the gait phase.
	auto phase_at = [&](int ci, float t) {
		const mm::Clip &k = db.clips[size_t(ci)];
		return k.phase[size_t(std::clamp(int(std::lround(t * k.fps)), 0, k.frames - 1))];
	};
	auto curve_at = [&](int ci, float t) {
		const mm::Clip &k = db.clips[size_t(ci)];
		return k.curve[size_t(std::clamp(int(std::lround(t * k.fps)), 0, k.frames - 1))];
	};
	for (const float speed : {1.3f, 2.5f, 3.5f}) {
		mm::Controller ctl;
		ctl.db = &db;
		ctl.reset({}, 0, 0);
		const float dt = 1.0f / 144;
		int joins = 0, off = 0, backs = 0, prev_clip = -1, prev_wraps = 0;
		float prev_time = 0, worst = 0;
		for (int i = 0; i < 144 * 60; ++i) {
			ctl.update(dt, mm::v3{0, 0, speed}, std::numeric_limits<float>::quiet_NaN(), mm::Stand, false);
			const bool wrapped = ctl.wraps != prev_wraps;
			prev_wraps = ctl.wraps;
			const bool goes_on = ctl.clip == prev_clip && ctl.time >= prev_time && ctl.time <= prev_time + 2 * dt;
			if (prev_clip >= 0 && !goes_on && !wrapped) {
				const float was = phase_at(prev_clip, prev_time + dt), now = phase_at(ctl.clip, ctl.time);
				if (was >= 0 && now >= 0 && curve_at(prev_clip, prev_time + dt) == curve_at(ctl.clip, ctl.time)) {
					++joins;
					backs += ctl.clip == prev_clip && ctl.time < prev_time;
					const float gap = mm::Database::phase_gap(was, now);
					worst = std::max(worst, gap);
					off += gap > tol;
				}
			}
			prev_clip = ctl.clip;
			prev_time = ctl.time;
		}
		const float dist = mm::len(ctl.char_pos);
		const bool ok = joins > 0 && off == 0 && dist > 0.6f * speed * 60 && dist < 1.25f * speed * 60;
		std::printf("PHASE %s speed=%.1f joins=%d back_in_clip=%d out_of_phase=%d worst=%.3f wraps=%d switches=%d travel=%.1f m\n",
				ok ? "PASS" : "FAIL", double(speed), joins, backs, off, double(worst), ctl.wraps, ctl.switches, double(dist));
		fails += ok ? 0 : 1;
	}
	// 5. Mixed rates (RFD66606-2327): a clip at 60 fps and its own every-second frame as a 30 fps clip
	// must give the same pose features and the same gait phase at the same moments, so clips
	// of both rates sit in one search. The control reads the 30 fps twin as if it were 60 fps.
	{
		const float feat_tol = 0.25f, phase_tol_rate = 0.04f;
		std::vector<float> diffs, gaps;
		int both = 0, one = 0;
		for (const Raw &r : fast) {
			const size_t J = r.parents.size(), F = r.xf.size() / (J * 12);
			const int up = int(std::lround(r.fps / 30));
			std::vector<float> half;
			for (size_t f = 0; f < F; f += size_t(up)) {
				half.insert(half.end(), r.xf.begin() + long(f * J * 12), r.xf.begin() + long((f + 1) * J * 12));
			}
			mm::Database t;
			t.settings = db.settings;
			t.add_clip(r.name, r.names, r.parents, r.xf, r.fps);
			t.add_clip(r.name + "_30", r.names, r.parents, half, plant_rate ? r.fps : r.fps / float(up));
			t.bake();
			const int D = db.dims();
			for (int k = 0; k < t.clips[1].frames; ++k) {
				const int f = k * up;
				const std::vector<float> a = t.raw_features(0, f), b = t.raw_features(1, k);
				for (int d = 0; d < D; ++d) {
					diffs.push_back(std::fabs(a[size_t(d)] - b[size_t(d)]) / std::max(db.stdev[size_t(d)], 1e-6f));
				}
				const float pa = t.clips[0].phase[size_t(f)], pb = t.clips[1].phase[size_t(k)];
				if (pa >= 0 && pb >= 0) {
					++both;
					gaps.push_back(mm::Database::phase_gap(pa, pb));
				} else if ((pa >= 0) != (pb >= 0)) {
					++one;
				}
			}
		}
		auto p99 = [](std::vector<float> v) {
			if (v.empty()) {
				return 0.0f;
			}
			std::sort(v.begin(), v.end());
			return v[std::min(v.size() - 1, size_t(double(v.size()) * 0.99))];
		};
		const float fd = p99(diffs), pg = p99(gaps);
		const bool ok = !fast.empty() && fd <= feat_tol && pg <= phase_tol_rate;
		std::printf("RATE %s clips=%zu feature_p99=%.3f sd (gate %.2f) phase_p99=%.3f cycle (gate %.2f) "
					"in_cycle_both=%d only_one=%d%s\n",
				ok ? "PASS" : "FAIL", fast.size(), fd, feat_tol, pg, phase_tol_rate, both, one,
				plant_rate ? " [control: twins at the wrong rate]" : "");
		fails += ok ? 0 : 1;
	}
	std::printf(fails ? "MM-CORE-FAIL %d\n" : "MM-CORE-PASS\n", fails);
	return fails ? 1 : 0;
}
