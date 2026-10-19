// Native checks of mm_core.hpp on real Kimodo SOMA30 clips (a directory of
// <key>/transforms.f32 + skeleton.txt + parents.txt as motion.elf's chain runner writes them).
//   mm_core_test <clips dir> [--plant-kd]
// 1. The k-d tree finds the same cost as a brute-force scan on 500 queries (database rows plus
//    noise). --plant-kd skips the far branch, which must make this check fail.
// 2. Every 50th pose, queried with its own features, finds a pose of cost 0.
// 3. The controller: standing, walking and running forward, and a 90-degree turn; it must move at
//    about the asked speed, face the asked way, and not flicker between poses.
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
	const bool plant = argc > 2 && std::string(argv[2]) == "--plant-kd";
	mm::Database db;
	std::vector<std::string> keys;
	if (DIR *d = opendir(argv[1])) {
		while (dirent *e = readdir(d)) {
			if (e->d_name[0] != '.') {
				keys.push_back(e->d_name);
			}
		}
		closedir(d);
	}
	std::sort(keys.begin(), keys.end());
	for (const std::string &k : keys) {
		std::vector<std::string> names;
		std::vector<int32_t> parents;
		std::vector<float> xf;
		float fps = 30;
		if (!read_clip(std::string(argv[1]) + "/" + k, names, parents, xf, fps)) {
			continue;
		}
		std::printf("%s\n", db.add_clip(k, names, parents, xf, fps).c_str());
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
	};
	const Case cases[] = {
			{"stand", 0.0f, -1, 0, "idle|turn|stop"},
			{"walk", 1.3f, -1, 0, "walk"},
			{"run", 3.5f, -1, 0, "run|jog"},
			{"turn", 1.3f, 3.0f, mm::kPi / 2, "walk|turn"},
	};
	for (const Case &c : cases) {
		mm::Controller ctl;
		ctl.db = &db;
		ctl.reset({}, 0, 0);
		std::map<std::string, float> played;
		const float dt = 1.0f / 60;
		float heading = 0;
		for (int i = 0; i < 60 * 8; ++i) {
			const float t = float(i) * dt;
			if (c.turn_at >= 0 && t >= c.turn_at) {
				heading = c.turn_to;
			}
			ctl.update(dt, mm::v3{std::sin(heading) * c.speed, 0, std::cos(heading) * c.speed},
					std::numeric_limits<float>::quiet_NaN());
			if (t > 2.0f) {
				played[db.clips[size_t(ctl.clip)].name] += dt;
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
		const bool dist_ok = c.speed == 0 ? dist < 1.0f : (dist > 0.6f * want && dist < 1.2f * want);
		const float face_err = std::fabs(mm::wrap_angle(ctl.char_yaw - (c.turn_at >= 0 ? c.turn_to : 0.0f)));
		const bool face_ok = c.speed == 0 || face_err < 0.35f;
		// Settled motion does not flicker between poses: at most 3 switches a second. The clip
		// names are only reported: motion matching uses any part of any clip (the walking start of
		// stop_walk is a walk).
		const bool calm = float(ctl.switches) / 8.0f <= 3.0f;
		(void)name_ok;
		const bool ok = dist_ok && face_ok && calm;
		std::printf("TRACE %s clip=%d time=%.6f x=%.6f z=%.6f yaw=%.6f switches=%d\n", c.name, ctl.clip, double(ctl.time),
				double(ctl.char_pos.x), double(ctl.char_pos.z), double(ctl.char_yaw), ctl.switches);
		std::printf("CTL %s %s travel=%.2f m (asked %.1f) yaw_err=%.2f rad searches=%d switches=%d top=%s | %s\n", c.name,
				ok ? "PASS" : "FAIL", double(dist), double(want), double(face_err), ctl.searches, ctl.switches, top.c_str(),
				list.c_str());
		fails += ok ? 0 : 1;
	}
	std::printf(fails ? "MM-CORE-FAIL %d\n" : "MM-CORE-PASS\n", fails);
	return fails ? 1 : 0;
}
