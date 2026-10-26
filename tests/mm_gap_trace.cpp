// The clip switches that still jump, now that walk mode plays the picked clip alone (no fade).
//   mm_gap_trace <clips dir[:dir...]> <switches.tsv>
// Every run settles 3 s in one state, then asks for another state at a heading for 3 s. States:
// idle, walk, run, crouch idle, crouch walk, crawl idle, crawl; headings 0, 45, 90, 135 and 180
// degrees from the settled facing. Each frame the shown pose is the clip's frame under the
// character: bone world = char_pos + rotate_y(bone in the clip's root frame, char_yaw), hips yaw =
// char_yaw + (the clip's hips yaw - the clip's root yaw). A switch frame whose bone move or hips
// turn is larger than the largest ordinary frame of all runs is a gap. The TSV lists every switch;
// stdout groups the gaps by asked turn, speed and mode.
#include "mm_core.hpp"

#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <map>
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

// The hips' world yaw per frame: full 3x3 rotations composed down the chain, forward = column z.
static std::vector<float> hips_yaw_track(const std::vector<std::string> &names, const std::vector<int32_t> &parents,
		const std::vector<float> &xf) {
	const size_t J = names.size(), F = xf.size() / (J * 12);
	const size_t hips = size_t(std::find(names.begin(), names.end(), std::string("Hips")) - names.begin());
	std::vector<float> out;
	std::vector<mm::m3> gr(J);
	for (size_t f = 0; f < F; ++f) {
		for (size_t j = 0; j < J; ++j) {
			mm::m3 r;
			std::copy(&xf[(f * J + j) * 12], &xf[(f * J + j) * 12] + 9, r.m);
			gr[j] = parents[j] < 0 ? r : mm::mul(gr[size_t(parents[j])], r);
		}
		out.push_back(std::atan2(gr[hips].m[2], gr[hips].m[8]));
	}
	return out;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: mm_gap_trace <clips dir[:dir...]> <switches.tsv>\n");
		return 2;
	}
	mm::Database db;
	std::vector<std::vector<float>> hips_yaw;
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
		for (const std::string &path : here) {
			std::vector<std::string> names;
			std::vector<int32_t> parents;
			std::vector<float> xf;
			float fps = 30;
			if (!read_clip(path, names, parents, xf, fps)) {
				continue;
			}
			const bool loop = std::ifstream(path + "/loop.txt").good();
			db.add_clip(path.substr(path.find_last_of('/') + 1), names, parents, xf, fps, loop);
			hips_yaw.push_back(hips_yaw_track(names, parents, xf));
			std::ifstream pf(path + "/phase.f32", std::ios::binary);
			if (pf) {
				pf.seekg(0, std::ios::end);
				std::vector<float> ph(size_t(pf.tellg()) / 4);
				pf.seekg(0);
				pf.read(reinterpret_cast<char *>(ph.data()), long(ph.size() * 4));
				db.set_phase(int(db.clips.size()) - 1, ph);
			}
		}
		s = e + 1;
	}
	std::printf("%s\n", db.bake().c_str());
	struct State {
		const char *name;
		float speed;
		int mode;
	};
	const State states[] = {{"idle", 0, mm::Stand}, {"walk", 1.3f, mm::Stand}, {"run", 3.5f, mm::Stand},
			{"crouch_idle", 0, mm::Crouch}, {"crouch_walk", 0.8f, mm::Crouch}, {"crawl_idle", 0, mm::Crawl},
			{"crawl", 0.4f, mm::Crawl}};
	const int turns[] = {0, 45, 90, 135, 180};
	const size_t B = db.bone_names.size();
	struct Shown {
		mm::v3 bone[8];
		float hips;
	};
	auto shown = [&](const mm::Controller &ctl) {
		const mm::Clip &c = db.clips[size_t(ctl.clip)];
		const int f = std::clamp(int(std::lround(ctl.time * c.fps)), 0, c.frames - 1);
		Shown s{};
		for (size_t b = 0; b < B; ++b) {
			const mm::v3 l = mm::to_local(c.bones[size_t(f) * B + b], c.root[size_t(f)], c.yaw[size_t(f)]);
			s.bone[b] = ctl.char_pos + mm::rotate_y(l, ctl.char_yaw);
		}
		s.hips = ctl.char_yaw + hips_yaw[size_t(ctl.clip)][size_t(f)] - c.yaw[size_t(f)];
		return s;
	};
	struct Switch {
		std::string from, to, from_clip, to_clip;
		int turn, mode;
		float t, speed, cost, move_cm, turn_deg;
		int from_f, to_f;
	};
	std::vector<Switch> sw;
	float ord_move = 0, ord_turn = 0;
	const float dt = 1.0f / 60;
	for (const State &a : states) {
		for (const State &b : states) {
			for (const int turn : turns) {
				if (b.speed == 0 && turn != 0) {
					continue; // a still state asks for no heading
				}
				mm::Controller ctl;
				ctl.db = &db;
				ctl.reset({}, 0, 0);
				int prev_clip = -1, prev_wraps = 0;
				float prev_time = 0;
				Shown prev{};
				const float to = float(turn) * mm::kPi / 180;
				for (int i = 0; i < 60 * 6; ++i) {
					const float t = float(i) * dt;
					const bool second = t >= 3;
					const State &st = second ? b : a;
					const float h = second ? to : 0.0f;
					const float sp = st.speed;
					ctl.update(dt, mm::v3{std::sin(h) * sp, 0, std::cos(h) * sp}, std::numeric_limits<float>::quiet_NaN(),
							st.mode, false);
					const Shown now = shown(ctl);
					const bool wrapped = ctl.wraps != prev_wraps;
					prev_wraps = ctl.wraps;
					const bool goes_on = ctl.clip == prev_clip && ctl.time >= prev_time && ctl.time <= prev_time + 2 * dt;
					if (prev_clip >= 0 && !wrapped) {
						float mv = 0;
						for (size_t k = 0; k < B; ++k) {
							mv = std::max(mv, mm::len(now.bone[k] - prev.bone[k]) * 100);
						}
						const float tn = std::fabs(mm::wrap_angle(now.hips - prev.hips)) * 180 / mm::kPi;
						if (goes_on) {
							ord_move = std::max(ord_move, mv);
							ord_turn = std::max(ord_turn, tn);
						} else {
							const mm::Clip &pc = db.clips[size_t(prev_clip)], &nc = db.clips[size_t(ctl.clip)];
							const float asked = std::fabs(mm::wrap_angle(h - prev.hips)) * 180 / mm::kPi;
							sw.push_back({a.name, b.name, pc.name, nc.name, int(std::lround(asked)), st.mode, t, sp, ctl.last_cost,
									mv, tn, int(std::lround(prev_time * pc.fps)), int(std::lround(ctl.time * nc.fps))});
						}
					}
					prev = now;
					prev_clip = ctl.clip;
					prev_time = ctl.time;
				}
			}
		}
	}
	std::ofstream out(argv[2]);
	out << "from\tto\tt\tasked_turn_deg\tspeed\tmode\tcost\tfrom_clip\tfrom_frame\tto_clip\tto_frame\tmove_cm\thips_turn_deg\tgap\n";
	std::map<std::string, std::pair<int, float>> groups;
	int gaps = 0;
	for (const Switch &s : sw) {
		const bool gap = s.move_cm > ord_move || s.turn_deg > ord_turn;
		gaps += gap;
		out << s.from << '\t' << s.to << '\t' << s.t << '\t' << s.turn << '\t' << s.speed << '\t' << s.mode << '\t' << s.cost
			<< '\t' << s.from_clip << '\t' << s.from_f << '\t' << s.to_clip << '\t' << s.to_f << '\t' << s.move_cm << '\t'
			<< s.turn_deg << '\t' << (gap ? 1 : 0) << '\n';
		if (gap) {
			const int bucket = int(std::lround(float(s.turn) / 45)) * 45;
			const char *kind = s.speed == 0 ? "still" : s.speed < 2 ? "slow" : "fast";
			const char *mode = s.mode == mm::Crouch ? "crouch" : s.mode == mm::Crawl ? "crawl" : "stand";
			auto &g = groups[std::string(mode) + " " + kind + " turn" + std::to_string(bucket)];
			g.first += 1;
			g.second = std::max(g.second, std::max(s.move_cm / ord_move, s.turn_deg / ord_turn));
		}
	}
	std::printf("GAP-TRACE switches=%zu gaps=%d ordinary_max move=%.1f cm hips_turn=%.1f deg\n", sw.size(), gaps,
			double(ord_move), double(ord_turn));
	for (auto &[k, v] : groups) {
		std::printf("GAP %s count=%d worst=%.1fx ordinary\n", k.c_str(), v.first, double(v.second));
	}
	return 0;
}
