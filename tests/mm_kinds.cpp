// The motion matcher's kind for every frame of every clip: one <out>/<clip>.kinds file per clip,
// one digit per line (0 stand, 1 crouch, 2 crawl, 4 air, 5 between), as mm_core.hpp's classify()
// labels them with all the clips loaded together (the standing head height is shared).
//   mm_kinds <out dir> <clip dir>...
#include "mm_core.hpp"

#include <cstdio>
#include <fstream>
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
	if (argc < 3) {
		std::fprintf(stderr, "usage: mm_kinds <out dir> <clip dir>...\n");
		return 2;
	}
	mm::Database db;
	for (int i = 2; i < argc; ++i) {
		std::vector<std::string> names;
		std::vector<int32_t> parents;
		std::vector<float> xf;
		float fps = 30;
		const std::string path = argv[i];
		if (!read_clip(path, names, parents, xf, fps)) {
			continue;
		}
		const bool loop = std::ifstream(path + "/loop.txt").good();
		db.add_clip(path.substr(path.find_last_of('/') + 1), names, parents, xf, fps, loop);
	}
	std::printf("%s\n", db.bake().c_str());
	for (const mm::Clip &c : db.clips) {
		std::ofstream o(std::string(argv[1]) + "/" + c.name + ".kinds");
		for (int f = 0; f < c.frames; ++f) {
			o << int(c.kind[size_t(f)]) << '\n';
		}
	}
	std::printf("KINDS clips=%zu standing_head=%.3f\n", db.clips.size(), double(db.standing_head));
	return 0;
}
