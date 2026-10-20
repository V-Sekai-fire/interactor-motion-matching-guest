// Remake a motion set's steady gaits as loops (src/mm_loop.hpp, RFD 2327).
//   make_loops <set dir> <list>            cut each listed clip in place and write its loop.txt
//   make_loops <set dir> <list> --check    measure the loops already cut; nothing is written
//   make_loops <set dir> <list> --uncut    control: play each listed clip uncut as a loop
// A clip whose seam is over one ordinary frame step is not cut and stays a one-shot. Prints
// LOOPS-OK when every seam it measured is within the gate, else LOOPS-FAIL.
#include "mm_loop.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

static bool read_lines(const std::string &path, std::vector<std::string> &out) {
	std::ifstream in(path);
	std::string line;
	while (std::getline(in, line)) {
		if (!line.empty()) {
			out.push_back(line);
		}
	}
	return bool(in.eof());
}

static bool read_floats(const std::string &path, std::vector<float> &xf) {
	std::ifstream tr(path, std::ios::binary);
	if (!tr) {
		return false;
	}
	tr.seekg(0, std::ios::end);
	xf.resize(size_t(tr.tellg()) / 4);
	tr.seekg(0);
	tr.read(reinterpret_cast<char *>(xf.data()), long(xf.size() * 4));
	return true;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: make_loops <set dir> <list> [--check | --uncut]\n");
		return 2;
	}
	const std::string set = argv[1];
	const std::string flag = argc > 3 ? argv[3] : "";
	std::vector<std::string> names;
	read_lines(argv[2], names);
	int bad = 0, measured = 0;
	for (const std::string &n : names) {
		const std::string dir = set + "/" + n;
		std::vector<std::string> sk, pl;
		std::vector<int32_t> parents;
		std::vector<float> xf;
		if (!read_lines(dir + "/skeleton.txt", sk) || !read_lines(dir + "/parents.txt", pl) ||
				!read_floats(dir + "/transforms.f32", xf)) {
			std::printf("LOOP-MISSING %s\n", n.c_str());
			++bad;
			continue;
		}
		for (const std::string &p : pl) {
			parents.push_back(int32_t(std::stoi(p)));
		}
		float fps = 30;
		for (const std::string &l : sk) {
			if (l.rfind("frames ", 0) == 0) {
				std::istringstream s(l);
				std::string w;
				int frames = 0;
				s >> w >> frames >> w >> fps;
			}
		}
		const int fade = std::max(1, int(std::lround(0.2f * fps)));
		const bool cut = std::ifstream(dir + "/loop.txt").good();
		if (flag == "--check" || flag == "--uncut") {
			if (flag == "--check" && !cut) {
				std::printf("LOOP-ONESHOT %s (not cut)\n", n.c_str());
				continue;
			}
			const mm::Frames c = mm::frames_of(xf, parents);
			const float seam = mm::seam_step(c, flag == "--uncut" ? 0 : fade);
			// A cut loop is held to its source clip's ordinary step, kept in its loop.txt.
			float ord = mm::ordinary_step(c);
			std::vector<std::string> lt;
			if (cut && read_lines(dir + "/loop.txt", lt) && !lt.empty()) {
				const size_t at = lt[0].find("ordinary_deg ");
				ord = at == std::string::npos ? ord : std::stof(lt[0].substr(at + 13));
			}
			++measured;
			bad += seam > ord;
			std::printf("LOOP-%s %s seam %.2f deg ordinary %.2f deg\n", seam <= ord ? "SEAM-OK" : "SEAM-OVER", n.c_str(),
					seam, ord);
			continue;
		}
		if (cut) {
			std::printf("LOOP-KEPT %s (already cut)\n", n.c_str());
			continue;
		}
		const mm::LoopCut lc = mm::cut_loop(xf, parents, fps);
		++measured;
		if (!lc.ok()) {
			std::printf("LOOP-ONESHOT %s seam %.2f deg over ordinary %.2f deg\n", n.c_str(), lc.seam_deg, lc.ordinary_deg);
			continue;
		}
		const std::vector<float> out = mm::floats_of(lc.out);
		std::ofstream(dir + "/transforms.f32", std::ios::binary)
				.write(reinterpret_cast<const char *>(out.data()), long(out.size() * 4));
		std::ofstream s(dir + "/skeleton.txt");
		for (const std::string &l : sk) {
			if (l.rfind("frames ", 0) == 0) {
				std::istringstream in(l);
				std::string w, rest;
				int frames = 0;
				in >> w >> frames;
				std::getline(in, rest);
				s << "frames " << lc.out.F << rest << "\n";
			} else {
				s << l << "\n";
			}
		}
		std::ofstream(dir + "/loop.txt") << "loop from " << lc.from << " to " << lc.to << " of " << (xf.size() / (parents.size() * 12))
										 << " seam_deg " << lc.seam_deg << " ordinary_deg " << lc.ordinary_deg << "\n";
		std::printf("LOOP-CUT %s frames %d..%d (%.2f s) seam %.2f deg ordinary %.2f deg\n", n.c_str(), lc.from, lc.to,
				float(lc.to - lc.from) / fps, lc.seam_deg, lc.ordinary_deg);
	}
	std::printf("%s measured=%d over=%d\n", bad ? "LOOPS-FAIL" : "LOOPS-OK", measured, bad);
	return bad ? 1 : 0;
}
