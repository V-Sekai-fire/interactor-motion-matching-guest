// Motion matching without Godot: a port of godot-motion-matching (Guilherme Sousa and
// contributors, MIT; V-Sekai/godot-motion-matching) for a godot-sandbox guest.
//
// Ported: the pose database (MMAnimationLibrary::bake_data), the trajectory feature
// (MMTrajectoryFeature: future and past root points with facing, in the pose's root frame), the
// bone feature (MMBoneDataFeature: bone positions in the root frame), standard normalisation and
// per-dimension weights (MMFeature), the weighted k-d tree search (KDTree), the critically damped
// velocity spring and trajectory prediction (Spring, MMCharacter::_generate_trajectory /
// _update_history). Changed for the guest: clips arrive as SOMA30 transforms (motion.elf's layout)
// instead of Godot Animation tracks; the root is the hips projected to the floor with a smoothed
// facing; a bone velocity feature is added (Holden, "Learned Motion Matching"); the character
// follows the clip's root motion and is pulled toward the simulation (no physics body).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace mm {

struct v3 {
	float x = 0, y = 0, z = 0;
};
inline v3 operator+(v3 a, v3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline v3 operator-(v3 a, v3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline v3 operator*(v3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float len(v3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }

// Row-major 3x3 acting on column vectors, as motion.elf writes it.
struct m3 {
	float m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};
inline m3 mul(const m3 &a, const m3 &b) {
	m3 r;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			r.m[i * 3 + j] = a.m[i * 3] * b.m[j] + a.m[i * 3 + 1] * b.m[3 + j] + a.m[i * 3 + 2] * b.m[6 + j];
		}
	}
	return r;
}
inline v3 mul(const m3 &a, v3 v) {
	return {a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z, a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z,
		a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z};
}

constexpr float kPi = 3.14159265358979f;
inline float wrap_angle(float a) {
	while (a > kPi) {
		a -= 2 * kPi;
	}
	while (a < -kPi) {
		a += 2 * kPi;
	}
	return a;
}
// A world point in the frame of a root at `origin` facing `yaw` (yaw 0 faces +Z, atan2(x, z)).
inline v3 to_local(v3 p, v3 origin, float yaw) {
	const v3 d = p - origin;
	const float c = std::cos(yaw), s = std::sin(yaw);
	return {c * d.x - s * d.z, d.y, s * d.x + c * d.z};
}
inline v3 rotate_y(v3 d, float yaw) {
	const float c = std::cos(yaw), s = std::sin(yaw);
	return {c * d.x + s * d.z, d.y, -s * d.x + c * d.z};
}

// Spring (spring.hpp): Holden's critically damped spring on a velocity.
inline float fast_negexp(float x) { return 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x); }
inline float halflife_to_damping(float halflife, float eps = 1e-5f) { return (4.0f * 0.69314718056f) / (halflife + eps); }
inline void simple_spring_damper_exact(v3 &x, v3 &v, v3 goal, float halflife, float dt) {
	const float y = halflife_to_damping(halflife) / 2.0f;
	const v3 j0 = x - goal;
	const v3 j1 = v + j0 * y;
	const float eydt = fast_negexp(y * dt);
	x = (j0 + j1 * dt) * eydt + goal;
	v = (v - j1 * (y * dt)) * eydt;
}
inline float damp_factor(float halflife, float dt) { return 1.0f - fast_negexp((0.69314718056f * dt) / (halflife + 1e-8f)); }

// The k-d tree (algo/kd_tree.cpp), unchanged apart from std types.
class KDTree {
public:
	bool skip_far = false; // test control only: a broken search that never visits the far side
	// The tree holds the rows listed in `rows` (all rows when empty).
	KDTree(const float *data, int dim, int count, std::vector<int> rows = {}) : dim_(dim) {
		if (rows.empty()) {
			rows.resize(static_cast<size_t>(count));
			std::iota(rows.begin(), rows.end(), 0);
		}
		root_ = build(data, rows.data(), int(rows.size()), 0);
	}
	int search_nn(const float *data, const float *query, const std::vector<float> &w, int *visited = nullptr) const {
		float best = std::numeric_limits<float>::max();
		int best_index = -1;
		int count = 0;
		search(data, query, w, root_.get(), best, best_index, 0, count);
		if (visited) {
			*visited = count;
		}
		return best_index;
	}

private:
	struct Node {
		int index = -1;
		std::unique_ptr<Node> child[2];
	};
	int dim_;
	std::unique_ptr<Node> root_;
	std::unique_ptr<Node> build(const float *data, int *idx, int n, int depth) {
		if (n <= 0) {
			return nullptr;
		}
		auto node = std::make_unique<Node>();
		const int axis = depth % dim_;
		const int mid = (n - 1) / 2;
		std::nth_element(idx, idx + mid, idx + n,
				[&](int a, int b) { return data[size_t(a) * size_t(dim_) + size_t(axis)] < data[size_t(b) * size_t(dim_) + size_t(axis)]; });
		node->index = idx[mid];
		node->child[0] = build(data, idx, mid, depth + 1);
		node->child[1] = build(data, idx + mid + 1, n - mid - 1, depth + 1);
		return node;
	}
	void search(const float *data, const float *q, const std::vector<float> &w, const Node *node, float &best,
			int &best_index, int depth, int &count) const {
		if (!node) {
			return;
		}
		++count;
		const float *p = data + size_t(node->index) * size_t(dim_);
		float d2 = 0;
		for (int i = 0; i < dim_; ++i) {
			const float d = p[i] - q[i];
			d2 += d * d * w[size_t(i)];
		}
		if (d2 < best) {
			best = d2;
			best_index = node->index;
		}
		const int axis = depth % dim_;
		const float diff = q[axis] - p[axis];
		const Node *nearc = diff < 0 ? node->child[0].get() : node->child[1].get();
		const Node *farc = diff < 0 ? node->child[1].get() : node->child[0].get();
		search(data, q, w, nearc, best, best_index, depth + 1, count);
		if (!skip_far && diff * diff * w[size_t(axis)] < best) {
			search(data, q, w, farc, best, best_index, depth + 1, count);
		}
	}
};

struct Clip {
	std::string name;
	float fps = 30;
	int frames = 0;
	// Per frame, from forward kinematics: the root (hips on the floor) and its smoothed facing,
	// and the feature bones in world space.
	std::vector<v3> root;
	std::vector<float> yaw;
	std::vector<v3> bones; // frames x bone_count
	std::vector<float> head_y;
	// Per frame, the heights of the joints that can carry the body besides the feet (the knees
	// and the hands, found by role in add_clip): frames x supports.
	int supports = 0;
	std::vector<float> support_y;
	// Per frame, from bake(): the pose's kind (see Mode) and, for a jump take-off, the frame where
	// the jump lands.
	std::vector<uint8_t> kind;
	std::vector<int> lands;
	// Per frame, from bake(): where the pose is in its gait cycle, 0 to 1 from one left-foot
	// strike to the next, or -1 outside a cycle (standing, a turn on the spot, a jump).
	std::vector<float> phase;
	// A learned phase (WalkTheDog's periodic autoencoder), when the host gives one: per frame the
	// manifold curve and the phase on it. It replaces the foot strikes; phases compare only on
	// one curve.
	std::vector<float> learned_phase;
	std::vector<int> curve;
	// A loop (a steady gait cut to whole strides): its last frame is its first moved on by one
	// pass, a turn of loop_yaw about the first root and a shift onto the last.
	bool loop = false;
	float loop_yaw = 0;
	float len_s() const { return float(frames - 1) / fps; }
};

// What the controller asks for. A pose's kind comes from its shape: the head's height against a
// standing head (stand above 82%, crouch down to 45%, crawl below), and Air when both feet are
// off the ground and no knee or hand is on it either (a crawl on hands and knees with the feet
// raised is carried, not flying). Jump is asked once: it plays a take-off pose and the jump
// through to landing.
// Between marks a short stretch of one kind inside another (a bend while walking, the stand-up of
// a crawl): it is played through but never matched.
enum Mode : uint8_t { Stand = 0, Crouch = 1, Crawl = 2, Jump = 3, Air = 4, Between = 5 };
constexpr int kModes = 4;

struct Settings {
	int future = 3;            // trajectory_point_count
	float future_dt = 1.0f / 3; // trajectory_delta_time
	int past = 1;              // history_point_count
	float past_dt = 1.0f / 3;  // history_delta_time
	float w_traj_pos = 1.0f, w_traj_facing = 1.0f, w_bone_pos = 1.0f, w_bone_vel = 1.0f;
	float halflife = 0.27f;        // the velocity spring
	float search_every = 0.1f;     // seconds between searches
	float same_clip_window = 0.2f; // a match this close to the playing frame is not a switch
	float switch_margin = 0.2f;    // the new pose must beat the playing pose's cost by this much
	float sync_halflife = 0.2f;    // the character is pulled toward the simulation
	float sync_clamp = 0.15f;      // and kept at most this far from it (metres)
	float facing_smooth = 0.25f;   // seconds either side for the root facing
	float vel_dt = 1.0f / 30;      // seconds either side for a bone velocity (one 30 fps frame)
	float end_margin = 1.0f;       // poses this close to a clip's end are never matched (a match plays at least this long)
	float crouch_head = 0.82f;     // a head below this share of a standing head is crouching
	float crawl_head = 0.45f;      // and below this, crawling
	float air_feet = 0.08f;        // both feet this far above the clip's ground: in the air (metres)
	float takeoff_from = 0.45f, takeoff_to = 0.1f; // take-off poses: this long before the feet leave
	float min_air = 0.15f;         // a shorter flight is a step, not a jump
	float min_run = 0.75f;         // a stand, crouch or crawl shorter than this is Between
	float phase_tol = 0.06f;       // a switch between two cycling poses keeps the phase this close (0: off)
	float strike_feet = 0.05f;     // a foot this near the clip's ground, and
	float strike_speed = 0.5f;     // this slow on the floor (m/s), is planted
	bool hand_support = true;      // knees and hands on the floor carry the body (classify)
	float min_cycle = 0.4f, max_cycle = 2.0f; // a gait cycle lasts this long
};

class Database {
public:
	Settings settings;
	std::vector<Clip> clips;
	std::vector<std::string> bone_names{"LeftFoot", "RightFoot", "Hips"};

	// A SOMA30 clip in motion.elf's layout: per frame, per joint, the parent-local row-major 3x3
	// then the translation (the root's is its position, the others their rest offset).
	std::string add_clip(const std::string &name, const std::vector<std::string> &names, const std::vector<int32_t> &parents,
			const std::vector<float> &xf, float fps, bool loop = false) {
		const size_t J = names.size();
		if (J == 0 || parents.size() != J || xf.size() % (J * 12) != 0) {
			return "FAIL joints " + std::to_string(J) + " parents " + std::to_string(parents.size()) + " floats " +
					std::to_string(xf.size());
		}
		const auto head_it = std::find(names.begin(), names.end(), std::string("Head"));
		if (head_it == names.end()) {
			return "FAIL no joint Head";
		}
		const size_t head = size_t(head_it - names.begin());
		std::vector<int> want;
		for (const std::string &b : bone_names) {
			const auto it = std::find(names.begin(), names.end(), b);
			if (it == names.end()) {
				return "FAIL no joint " + b;
			}
			want.push_back(int(it - names.begin()));
		}
		const int hips = want.back();
		const std::vector<int> sup = support_joints(parents, {want[0], want[1]}, int(head));
		Clip c;
		c.supports = int(sup.size());
		c.name = name;
		c.fps = fps;
		c.frames = int(xf.size() / (J * 12));
		c.loop = loop && c.frames > 1;
		std::vector<m3> gr(J);
		std::vector<v3> gp(J);
		std::vector<v3> fwd(size_t(c.frames));
		for (int f = 0; f < c.frames; ++f) {
			for (size_t j = 0; j < J; ++j) {
				const float *o = &xf[(size_t(f) * J + j) * 12];
				m3 r;
				std::copy(o, o + 9, r.m);
				const v3 t{o[9], o[10], o[11]};
				const int p = parents[j];
				if (p < 0) {
					gr[j] = r;
					gp[j] = t;
				} else {
					gr[j] = mul(gr[size_t(p)], r);
					gp[j] = gp[size_t(p)] + mul(gr[size_t(p)], t);
				}
			}
			c.root.push_back({gp[size_t(hips)].x, 0, gp[size_t(hips)].z});
			fwd[size_t(f)] = mul(gr[size_t(hips)], v3{0, 0, 1});
			for (int b : want) {
				c.bones.push_back(gp[size_t(b)]);
			}
			c.head_y.push_back(gp[head].y);
			for (int s : sup) {
				c.support_y.push_back(gp[size_t(s)].y);
			}
		}
		// The facing is averaged over facing_smooth seconds each side, whatever the clip's rate:
		// whole frames inside the window, and the frame just past each edge by the part of it
		// that falls inside, so 30 and 60 fps clips average the same span.
		const float hw = std::max(0.0f, settings.facing_smooth * fps);
		const int half = int(std::floor(hw)) + 1;
		const float edge = hw - std::floor(hw);
		const int L = c.frames - 1;
		if (c.loop) {
			c.loop_yaw = wrap_angle(std::atan2(fwd[size_t(L)].x, fwd[size_t(L)].z) - std::atan2(fwd[0].x, fwd[0].z));
		}
		for (int f = 0; f < c.frames; ++f) {
			v3 s;
			for (int k = f - half; k <= f + half; ++k) {
				const float w8 = std::abs(k - f) == half ? edge : 1.0f;
				// A loop's window runs on into the next pass (or back into the last), turned with it.
				if (c.loop && (k < 0 || k > L)) {
					const v3 w = k > L ? fwd[size_t(k - L)] : fwd[size_t(k + L)];
					s = s + rotate_y(v3{w.x, 0, w.z}, k > L ? c.loop_yaw : -c.loop_yaw) * w8;
				} else if (k >= 0 && k <= L) {
					s = s + v3{fwd[size_t(k)].x, 0, fwd[size_t(k)].z} * w8;
				}
			}
			c.yaw.push_back(std::atan2(s.x, s.z));
		}
		clips.push_back(std::move(c));
		return "ADDED " + name + " frames " + std::to_string(clips.back().frames);
	}

	int dims() const { return (settings.future + settings.past) * 3 + int(bone_names.size()) * 6; }
	int pose_count() const { return int(pose_clip.size()); }

	// MMAnimationLibrary::bake_data: one pose per frame of every clip, normalised per dimension.
	std::string bake() {
		data.clear();
		pose_clip.clear();
		pose_frame.clear();
		classify();
		const int D = dims();
		for (int ci = 0; ci < int(clips.size()); ++ci) {
			for (int f = 0; f < clips[size_t(ci)].frames; ++f) {
				std::vector<float> row = raw_features(ci, f);
				data.insert(data.end(), row.begin(), row.end());
				pose_clip.push_back(ci);
				pose_frame.push_back(f);
			}
		}
		const int N = pose_count();
		if (N == 0) {
			return "FAIL no poses";
		}
		mean.assign(size_t(D), 0);
		stdev.assign(size_t(D), 0);
		for (int i = 0; i < N; ++i) {
			for (int d = 0; d < D; ++d) {
				mean[size_t(d)] += data[size_t(i) * size_t(D) + size_t(d)];
			}
		}
		for (float &m : mean) {
			m /= float(N);
		}
		for (int i = 0; i < N; ++i) {
			for (int d = 0; d < D; ++d) {
				const float e = data[size_t(i) * size_t(D) + size_t(d)] - mean[size_t(d)];
				stdev[size_t(d)] += e * e;
			}
		}
		for (float &s : stdev) {
			s = std::sqrt(s / float(N));
		}
		for (int i = 0; i < N; ++i) {
			normalize(&data[size_t(i) * size_t(D)]);
		}
		weights = dimension_weights();
		// A match right before a clip's end would only play a few frames (Holden's range stops).
		eligible.clear();
		for (int i = 0; i < N; ++i) {
			const Clip &c = clips[size_t(pose_clip[size_t(i)])];
			if (c.loop || float(c.frames - 1 - pose_frame[size_t(i)]) / c.fps >= settings.end_margin) {
				eligible.push_back(i);
			}
		}
		tree = std::make_unique<KDTree>(data.data(), D, N, eligible);
		// One tree per mode: Stand, Crouch and Crawl hold the poses of that kind; Jump holds the
		// take-off poses.
		// A pose needs end_margin of its own kind ahead of it: a match at the end of a crouch inside
		// a walking clip would only crouch for a moment.
		std::vector<std::vector<int>> run_end(clips.size());
		for (size_t ci = 0; ci < clips.size(); ++ci) {
			const Clip &c = clips[ci];
			run_end[ci].assign(size_t(c.frames), c.frames - 1);
			for (int f = c.frames - 2; f >= 0; --f) {
				run_end[ci][size_t(f)] = c.kind[size_t(f)] == c.kind[size_t(f) + 1] ? run_end[ci][size_t(f) + 1] : f;
			}
		}
		std::string counts;
		for (int m = 0; m < kModes; ++m) {
			mode_rows[m].clear();
			for (int i = 0; i < N; ++i) {
				const int ci = pose_clip[size_t(i)];
				const Clip &c = clips[size_t(ci)];
				const int f = pose_frame[size_t(i)];
				const bool room = c.loop || float(run_end[size_t(ci)][size_t(f)] - f) / c.fps >= settings.end_margin;
				const bool in = m == Jump ? c.lands[size_t(f)] >= 0 : (c.kind[size_t(f)] == m && room);
				if (in) {
					mode_rows[m].push_back(i);
				}
			}
			mode_tree[m] = mode_rows[m].empty() ? nullptr : std::make_unique<KDTree>(data.data(), D, N, mode_rows[m]);
			counts += " " + std::to_string(mode_rows[m].size());
		}
		return "BAKED clips " + std::to_string(clips.size()) + " poses " + std::to_string(N) + " dims " + std::to_string(D) +
				" stand/crouch/crawl/jump" + counts;
	}

	// Runs of `kind` shorter than `secs`: to the kind just before (or after) when `to_neighbour`,
	// else to Between.
	static void relabel_short(Clip &c, uint8_t kind, float secs, bool to_neighbour) {
		const size_t F = c.kind.size();
		for (size_t a = 0; a < F;) {
			if (c.kind[a] != kind) {
				++a;
				continue;
			}
			size_t b = a;
			while (b + 1 < F && c.kind[b + 1] == kind) {
				++b;
			}
			if (float(b - a + 1) / c.fps < secs) {
				uint8_t to = Between;
				if (to_neighbour) {
					to = a > 0 ? c.kind[a - 1] : (b + 1 < F ? c.kind[b + 1] : uint8_t(Stand));
					if (to == kind) {
						to = Stand;
					}
				}
				for (size_t f = a; f <= b; ++f) {
					c.kind[f] = to;
				}
			}
			a = b + 1;
		}
	}

	// The joints besides the feet that can carry the body, found by role from the skeleton's tree
	// (no names): the knee is each foot's parent; the trunk is the chain from the root to the head;
	// an arm is a chain hanging off the trunk below the head that holds no foot, and its hand is
	// the first joint where the chain branches (into fingers), or its end when it never does.
	static std::vector<int> support_joints(const std::vector<int32_t> &parents, const std::vector<int> &feet, int head) {
		const size_t J = parents.size();
		std::vector<int> out;
		std::vector<char> trunk(J, 0), leg(J, 0);
		for (int j = head; j >= 0; j = parents[size_t(j)]) {
			trunk[size_t(j)] = 1;
		}
		for (int ft : feet) {
			out.push_back(parents[size_t(ft)]);
			for (int j = ft; j >= 0; j = parents[size_t(j)]) {
				leg[size_t(j)] = 1;
			}
		}
		std::vector<std::vector<int>> kids(J);
		for (size_t j = 0; j < J; ++j) {
			if (parents[j] >= 0) {
				kids[size_t(parents[j])].push_back(int(j));
			}
		}
		for (size_t j = 0; j < J; ++j) {
			const int p = parents[j];
			if (p < 0 || p == head || !trunk[size_t(p)] || trunk[j] || leg[j]) {
				continue;
			}
			int h = int(j);
			while (kids[size_t(h)].size() == 1) {
				h = kids[size_t(h)][0];
			}
			out.push_back(h);
		}
		return out;
	}

	// Each pose's kind, and the jumps' take-offs (see Mode).
	void classify() {
		std::vector<float> heads;
		for (const Clip &c : clips) {
			heads.insert(heads.end(), c.head_y.begin(), c.head_y.end());
		}
		std::sort(heads.begin(), heads.end());
		standing_head = heads.empty() ? 1.6f : heads[heads.size() * 9 / 10];
		const size_t B = bone_names.size();
		// Each clip's ground: its lowest foot's 10th percentile over the clip.
		std::vector<std::vector<float>> lows(clips.size());
		std::vector<float> grounds(clips.size());
		for (size_t ci = 0; ci < clips.size(); ++ci) {
			const Clip &c = clips[ci];
			const size_t F = size_t(c.frames);
			lows[ci].resize(F);
			for (size_t f = 0; f < F; ++f) {
				lows[ci][f] = std::min(c.bones[f * B + 0].y, c.bones[f * B + 1].y); // LeftFoot, RightFoot
			}
			std::vector<float> s = lows[ci];
			std::sort(s.begin(), s.end());
			grounds[ci] = s[s.size() / 10];
		}
		// How high above its clip's ground a joint that bears weight sits, from the feet: over every
		// frame that the foot rule calls Stand (head at standing height, a foot down), each foot
		// that is still (speed below strike_speed, the planted rule without its height test, since
		// the height is what is measured), its height above the ground. The bound is the 90th
		// percentile, the same 10% tail the ground drops. A knee or hand at most this high above
		// the ground is on the floor: a wrist or knee centre on the floor sits lower than an ankle
		// centre, so it is in contact whenever it is no higher than a planted ankle.
		std::vector<float> planted;
		for (size_t ci = 0; ci < clips.size(); ++ci) {
			const Clip &c = clips[ci];
			for (int f = 0; f < c.frames; ++f) {
				if (c.head_y[size_t(f)] / standing_head < settings.crouch_head || lows[ci][size_t(f)] > grounds[ci] + settings.air_feet) {
					continue;
				}
				for (size_t b = 0; b < 2; ++b) {
					const v3 v = bone_velocity(c, b, f);
					if (std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z) < settings.strike_speed) {
						planted.push_back(c.bones[size_t(f) * B + b].y - grounds[ci]);
					}
				}
			}
		}
		std::sort(planted.begin(), planted.end());
		contact_height = planted.empty() ? 0.0f : planted[planted.size() * 9 / 10];
		planted_samples = planted.size();
		for (size_t ci = 0; ci < clips.size(); ++ci) {
			Clip &c = clips[ci];
			const size_t F = size_t(c.frames);
			const std::vector<float> &low = lows[ci];
			const float ground = grounds[ci];
			const size_t S = size_t(c.supports);
			auto carried = [&](size_t f) {
				for (size_t s = 0; settings.hand_support && s < S; ++s) {
					if (c.support_y[f * S + s] <= ground + contact_height) {
						return true;
					}
				}
				return false;
			};
			c.kind.assign(F, Stand);
			c.lands.assign(F, -1);
			for (size_t f = 0; f < F; ++f) {
				const float h = c.head_y[f] / standing_head;
				if (low[f] > ground + settings.air_feet && !carried(f)) {
					c.kind[f] = Air;
				} else if (h < settings.crawl_head) {
					c.kind[f] = Crawl;
				} else if (h < settings.crouch_head) {
					c.kind[f] = Crouch;
				}
			}
			// A flight shorter than a jump is a running stride: it takes the kind before it.
			relabel_short(c, Air, settings.min_air, true);
			// Short stretches of one kind inside another are Between.
			for (uint8_t k : {uint8_t(Stand), uint8_t(Crouch), uint8_t(Crawl)}) {
				relabel_short(c, k, settings.min_run, false);
			}
			for (size_t a = 0; a < F;) {
				if (c.kind[a] != Air) {
					++a;
					continue;
				}
				size_t b = a;
				while (b + 1 < F && c.kind[b + 1] == Air) {
					++b;
				}
				const bool long_enough = float(b - a + 1) / c.fps >= settings.min_air;
				const bool lands_in_clip = float(F - 1 - b) / c.fps >= 0.2f;
				if (long_enough && lands_in_clip) {
					const long from = long(a) - long(std::lround(settings.takeoff_from * c.fps));
					const long to = long(a) - long(std::lround(settings.takeoff_to * c.fps));
					for (long f = std::max(0L, from); f <= to; ++f) {
						if (c.kind[size_t(f)] != Air) {
							c.lands[size_t(f)] = int(b);
						}
					}
				}
				a = b + 1;
			}
			if (c.learned_phase.size() == size_t(c.frames)) {
				c.phase = c.learned_phase;
			} else {
				c.curve.assign(size_t(c.frames), -1);
				cycle_phase(c, ground);
			}
		}
	}

	// Left-foot strikes (the foot comes to rest on the clip's ground) split the clip into gait
	// cycles; a loop's cycles run on across its seam into the next pass.
	void cycle_phase(Clip &c, float ground) const {
		const size_t B = bone_names.size();
		const int F = c.frames;
		const int n = c.loop ? 2 * F - 1 : F;
		auto at = [&](int f) { return c.bones[size_t(f % F) * B + 0]; }; // LeftFoot
		std::vector<int> strikes;
		bool was = true;
		for (int i = 0; i < n; ++i) {
			const int f = c.loop && i >= F ? i - (F - 1) : i;
			const v3 v = bone_velocity(c, 0, f);
			const bool planted = at(f).y <= ground + settings.strike_feet &&
					std::sqrt(v.x * v.x + v.z * v.z) < settings.strike_speed;
			if (planted && !was) {
				strikes.push_back(i);
			}
			was = planted;
		}
		std::vector<float> ph(size_t(n), -1.0f);
		for (size_t k = 0; k + 1 < strikes.size(); ++k) {
			const int a = strikes[k], b = strikes[k + 1];
			const float secs = float(b - a) / c.fps;
			if (secs < settings.min_cycle || secs > settings.max_cycle) {
				continue;
			}
			for (int i = a; i < b; ++i) {
				ph[size_t(i)] = float(i - a) / float(b - a);
			}
		}
		c.phase.assign(size_t(F), -1.0f);
		for (int i = 0; i < n; ++i) {
			const int f = c.loop && i >= F ? i - (F - 1) : i;
			if (ph[size_t(i)] >= 0 && (c.phase[size_t(f)] < 0 || i < F)) {
				c.phase[size_t(f)] = ph[size_t(i)];
			}
		}
		// Only walking, jogging and running cycle; a crouch or crawl keeps its own kind.
		for (int f = 0; f < F; ++f) {
			if (c.kind[size_t(f)] == Air || c.kind[size_t(f)] == Between) {
				c.phase[size_t(f)] = -1;
			}
		}
	}
	// The learned phase of a clip: frames x 4 (curve, phase, amplitude, frequency).
	std::string set_phase(int ci, const std::vector<float> &rows) {
		if (ci < 0 || ci >= int(clips.size()) || rows.size() != size_t(clips[size_t(ci)].frames) * 4) {
			return "FAIL phase rows " + std::to_string(rows.size());
		}
		Clip &c = clips[size_t(ci)];
		c.learned_phase.resize(size_t(c.frames));
		c.curve.resize(size_t(c.frames));
		for (int f = 0; f < c.frames; ++f) {
			c.curve[size_t(f)] = int(rows[size_t(f) * 4]);
			c.learned_phase[size_t(f)] = rows[size_t(f) * 4 + 1];
		}
		return "PHASE " + c.name + " frames " + std::to_string(c.frames);
	}
	// How far apart two phases are around the cycle (0 to 0.5); 0 when either is outside one.
	static float phase_gap(float a, float b) {
		if (a < 0 || b < 0) {
			return 0;
		}
		const float d = std::fabs(a - b);
		return std::min(d, 1 - d);
	}

	void normalize(float *row) const {
		for (size_t d = 0; d < mean.size(); ++d) {
			row[d] = (row[d] - mean[d]) / (stdev[d] + 1e-4f);
		}
	}

	// MMFeature::calculate_normalized_weight: a feature's weight shared over its dimensions.
	std::vector<float> dimension_weights() const {
		std::vector<float> w;
		const int T = settings.future + settings.past;
		for (int i = 0; i < T; ++i) {
			w.push_back(settings.w_traj_pos / float(2 * T));
			w.push_back(settings.w_traj_pos / float(2 * T));
			w.push_back(settings.w_traj_facing / float(T));
		}
		const int B = int(bone_names.size());
		for (int i = 0; i < B * 3; ++i) {
			w.push_back(settings.w_bone_pos / float(B * 3));
		}
		for (int i = 0; i < B * 3; ++i) {
			w.push_back(settings.w_bone_vel / float(B * 3));
		}
		return w;
	}

	// A pose's features before normalisation: future then past trajectory points (x, z, facing)
	// in the pose's root frame (MMTrajectoryFeature::bake_animation_pose, with its 0.1 s velocity
	// extrapolation past either end), then bone positions and velocities in the root frame.
	std::vector<float> raw_features(int ci, int f) const {
		const Clip &c = clips[size_t(ci)];
		const v3 origin = c.root[size_t(f)];
		const float yaw = c.yaw[size_t(f)];
		std::vector<float> out;
		auto point = [&](float t) {
			const float len_s = float(c.frames - 1) / c.fps;
			const float ct = std::clamp(t, 0.0f, len_s);
			v3 p = c.loop ? root_at(c, t) : sample_root(c, ct);
			if (!c.loop && (t > len_s || t < 0)) {
				const float e = 0.1f;
				const v3 vel = t > len_s ? (sample_root(c, len_s) - sample_root(c, len_s - e)) * (1 / e)
										 : (sample_root(c, 0) - sample_root(c, e)) * (1 / e);
				p = p + vel * (t > len_s ? t - len_s : -t);
			}
			const v3 l = to_local(p, origin, yaw);
			out.push_back(l.x);
			out.push_back(l.z);
			out.push_back(wrap_angle((c.loop ? yaw_at(c, t) : sample_yaw(c, ct)) - yaw));
		};
		const float t0 = float(f) / c.fps;
		for (int i = 1; i <= settings.future; ++i) {
			point(t0 + settings.future_dt * float(i));
		}
		for (int i = 1; i <= settings.past; ++i) {
			point(t0 - settings.past_dt * float(i));
		}
		bone_features(c, f, origin, yaw, out);
		return out;
	}

	void bone_features(const Clip &c, int f, v3 origin, float yaw, std::vector<float> &out) const {
		const size_t B = bone_names.size();
		for (size_t b = 0; b < B; ++b) {
			const v3 l = to_local(c.bones[size_t(f) * B + b], origin, yaw);
			out.push_back(l.x);
			out.push_back(l.y);
			out.push_back(l.z);
		}
		for (size_t b = 0; b < B; ++b) {
			const v3 l = rotate_y(bone_velocity(c, b, f), -yaw);
			out.push_back(l.x);
			out.push_back(l.y);
			out.push_back(l.z);
		}
	}

	// A bone's velocity at frame f: the central difference over vel_dt seconds each side, read
	// between frames, so it measures the same span at any frame rate (clamped at the clip ends).
	v3 bone_velocity(const Clip &c, size_t b, int f) const {
		const size_t B = bone_names.size();
		const float last = float(c.frames - 1) / c.fps, t = float(f) / c.fps;
		const float t0 = std::max(0.0f, t - settings.vel_dt), t1 = std::min(last, t + settings.vel_dt);
		if (t1 <= t0) {
			return {};
		}
		auto at = [&](float u) {
			const float x = std::clamp(u * c.fps, 0.0f, float(c.frames - 1));
			const int a = int(x), n = std::min(a + 1, c.frames - 1);
			const float w = x - float(a);
			return c.bones[size_t(a) * B + b] * (1 - w) + c.bones[size_t(n) * B + b] * w;
		};
		return (at(t1) - at(t0)) * (1 / (t1 - t0));
	}

	static v3 sample_root(const Clip &c, float t) {
		const float x = std::clamp(t * c.fps, 0.0f, float(c.frames - 1));
		const int a = int(x);
		const int b = std::min(a + 1, c.frames - 1);
		const float u = x - float(a);
		return c.root[size_t(a)] * (1 - u) + c.root[size_t(b)] * u;
	}
	static float sample_yaw(const Clip &c, float t) {
		const float x = std::clamp(t * c.fps, 0.0f, float(c.frames - 1));
		const int a = int(x);
		const int b = std::min(a + 1, c.frames - 1);
		const float u = x - float(a);
		return c.yaw[size_t(a)] + wrap_angle(c.yaw[size_t(b)] - c.yaw[size_t(a)]) * u;
	}

	// A loop's root and facing at any time: passes before or after this one are moved by the
	// loop's turn and shift. Other clips hold still past their ends.
	static int passes(const Clip &c, float t, float &in_pass) {
		const float L = c.len_s();
		const int n = int(std::floor(t / L));
		in_pass = std::clamp(t - float(n) * L, 0.0f, L);
		return n;
	}
	static v3 root_at(const Clip &c, float t) {
		if (!c.loop) {
			return sample_root(c, t);
		}
		float u = 0;
		int n = passes(c, t, u);
		v3 p = sample_root(c, u);
		const v3 a = c.root.front(), b = c.root.back();
		for (; n > 0; --n) {
			p = b + rotate_y(p - a, c.loop_yaw);
		}
		for (; n < 0; ++n) {
			p = a + rotate_y(p - b, -c.loop_yaw);
		}
		return p;
	}
	static float yaw_at(const Clip &c, float t) {
		if (!c.loop) {
			return sample_yaw(c, t);
		}
		float u = 0;
		const int n = passes(c, t, u);
		return wrap_angle(sample_yaw(c, u) + float(n) * c.loop_yaw);
	}
	// The best pose of a mode that `skip` does not refuse, by a full scan.
	template <typename Skip>
	int search_mode_except(int mode, const float *q, Skip skip) const {
		const int m = has_mode(mode) ? mode : int(Stand);
		int best = -1;
		float bc = std::numeric_limits<float>::max();
		for (int i : mode_rows[m]) {
			if (skip(pose_clip[size_t(i)], float(pose_frame[size_t(i)]) / clips[size_t(pose_clip[size_t(i)])].fps)) {
				continue;
			}
			const float c = cost(i, q);
			if (c < bc) {
				bc = c;
				best = i;
			}
		}
		return best;
	}

	float cost(int pose, const float *q) const {
		const int D = dims();
		const float *p = &data[size_t(pose) * size_t(D)];
		float s = 0;
		for (int d = 0; d < D; ++d) {
			const float e = p[d] - q[d];
			s += e * e * weights[size_t(d)];
		}
		return s;
	}
	int search_brute(const float *q) const {
		int best = -1;
		float bc = std::numeric_limits<float>::max();
		for (int i : eligible) {
			const float c = cost(i, q);
			if (c < bc) {
				bc = c;
				best = i;
			}
		}
		return best;
	}
	int search(const float *q, int *visited = nullptr) const { return tree->search_nn(data.data(), q, weights, visited); }
	// The best pose of a mode; a mode without poses falls back to Stand.
	int search_mode(int mode, const float *q) const {
		const KDTree *t = mode >= 0 && mode < kModes && mode_tree[mode] ? mode_tree[mode].get() : mode_tree[Stand].get();
		return t ? t->search_nn(data.data(), q, weights) : search(q);
	}
	bool has_mode(int mode) const { return mode >= 0 && mode < kModes && mode_tree[mode] != nullptr; }

	std::vector<float> data, mean, stdev, weights;
	std::vector<int> pose_clip, pose_frame, eligible;
	std::unique_ptr<KDTree> tree;
	std::vector<int> mode_rows[kModes];
	std::unique_ptr<KDTree> mode_tree[kModes];
	float standing_head = 1.6f;
	// From classify(): how high above its clip's ground a planted foot sits (90th percentile of
	// planted_samples), the bound for a knee or hand on the floor.
	float contact_height = 0.0f;
	size_t planted_samples = 0;
};

// MMCharacter without a physics body: a velocity spring predicts the trajectory; the character
// plays the matched clip, moves by the clip's root motion and is pulled toward the simulation.
struct Controller {
	const Database *db = nullptr;
	v3 sim_pos, sim_vel, sim_acc;
	float sim_yaw = 0;
	v3 char_pos;
	float char_yaw = 0;
	int clip = 0;
	float time = 0;
	float since_search = 1e9f;
	float last_cost = 0;
	int searches = 0, switches = 0, wraps = 0;
	// MMCharacter::_update_history: (clock, position, facing) of the simulation.
	struct Past {
		float t;
		v3 pos;
		float yaw;
	};
	std::vector<Past> history;
	float clock = 0;
	int mode = Stand;          // the kind of pose asked for (Stand, Crouch or Crawl)
	float locked_until = -1;   // a jump plays through to here in its clip before searching again
	int jumps = 0;

	void reset(v3 pos, float yaw, int start_clip) {
		sim_pos = char_pos = pos;
		sim_vel = sim_acc = {};
		sim_yaw = char_yaw = yaw;
		clip = start_clip;
		time = 0;
		since_search = 1e9f;
		history.clear();
		clock = 0;
		mode = Stand;
		locked_until = -1;
	}

	// One frame. desired: the ground velocity wanted (m/s, world); facing_yaw: NaN = face where
	// the simulation moves, else strafe facing that way.
	bool step(float dt, v3 desired, float facing_yaw) {
		const Settings &s = db->settings;
		desired.y = 0;
		// MMCharacter::_update_trajectory.
		simple_spring_damper_exact(sim_vel, sim_acc, desired, s.halflife, dt);
		sim_pos = sim_pos + sim_vel * dt;
		if (!std::isnan(facing_yaw)) {
			sim_yaw = facing_yaw;
		} else if (len(sim_vel) > 0.05f) {
			sim_yaw = std::atan2(sim_vel.x, sim_vel.z);
		}
		clock += dt;
		history.push_back({clock, sim_pos, sim_yaw});
		const float oldest = clock - s.past_dt * float(s.past) - 0.5f;
		size_t drop = 0;
		while (drop + 1 < history.size() && history[drop + 1].t < oldest) {
			++drop;
		}
		history.erase(history.begin(), history.begin() + long(drop));
		// Play on: root motion of the current clip moves the character.
		// A loop plays on through its end into its next pass, carrying the root with it.
		const Clip &c0 = db->clips[size_t(clip)];
		const float len_s = float(c0.frames - 1) / c0.fps;
		const float t1 = c0.loop ? time + dt : std::min(time + dt, len_s);
		const v3 d_local = to_local(Database::root_at(c0, t1), Database::root_at(c0, time), Database::yaw_at(c0, time));
		const float d_yaw = wrap_angle(Database::yaw_at(c0, t1) - Database::yaw_at(c0, time));
		char_pos = char_pos + rotate_y(v3{d_local.x, 0, d_local.z}, char_yaw);
		char_yaw = wrap_angle(char_yaw + d_yaw);
		time = t1;
		if (c0.loop && time >= len_s) {
			time -= len_s;
			++wraps;
		}
		// Pull the character toward the simulation, never farther than the clamp.
		const float k = damp_factor(s.sync_halflife, dt);
		char_pos = char_pos + (sim_pos - char_pos) * k;
		char_yaw = wrap_angle(char_yaw + wrap_angle(sim_yaw - char_yaw) * k);
		const v3 gap = char_pos - sim_pos;
		if (len(gap) > s.sync_clamp) {
			char_pos = sim_pos + gap * (s.sync_clamp / len(gap));
		}
		// Search every search_every seconds, or when the clip ran out; never inside a jump.
		since_search += dt;
		bool switched = false;
		const bool ran_out = !c0.loop && time >= len_s - 1e-4f;
		const bool locked = locked_until >= 0 && time < locked_until && !ran_out;
		if (!locked) {
			locked_until = -1;
		}
		if (!locked && (since_search >= s.search_every || ran_out)) {
			since_search = 0;
			switched = search();
		}
		return switched;
	}

	// MMCharacter::_generate_trajectory: future points from the spring, past from the history,
	// in the character's frame; bone features from the pose now playing.
	std::vector<float> query() const {
		const Settings &s = db->settings;
		std::vector<float> q;
		v3 p = sim_pos, v = sim_vel, a = sim_acc;
		float y = sim_yaw;
		const v3 goal_v = desired_cache;
		for (int i = 1; i <= s.future; ++i) {
			simple_spring_damper_exact(v, a, goal_v, s.halflife, s.future_dt);
			p = p + v * s.future_dt;
			if (std::isnan(facing_cache) && len(v) > 0.05f) {
				y = std::atan2(v.x, v.z);
			} else if (!std::isnan(facing_cache)) {
				y = facing_cache;
			}
			const v3 l = to_local(p, char_pos, char_yaw);
			q.push_back(l.x);
			q.push_back(l.z);
			q.push_back(wrap_angle(y - char_yaw));
		}
		for (int i = 1; i <= s.past; ++i) {
			const float want = clock - s.past_dt * float(i);
			v3 hp = sim_pos;
			float hy = sim_yaw;
			for (const Past &h : history) {
				if (h.t <= want) {
					hp = h.pos;
					hy = h.yaw;
				}
			}
			if (!history.empty() && history.front().t > want) {
				hp = history.front().pos;
				hy = history.front().yaw;
			}
			const v3 l = to_local(hp, char_pos, char_yaw);
			q.push_back(l.x);
			q.push_back(l.z);
			q.push_back(wrap_angle(hy - char_yaw));
		}
		const Clip &c = db->clips[size_t(clip)];
		const int f = std::clamp(int(std::lround(time * c.fps)), 0, c.frames - 1);
		db->bone_features(c, f, c.root[size_t(f)], c.yaw[size_t(f)], q);
		db->normalize(q.data());
		return q;
	}

	bool jumping() const { return locked_until >= 0; }

	// A jump: the take-off pose that fits best, then the jump plays through to its landing.
	bool jump() {
		if (jumping() || !db->has_mode(Jump)) {
			return false;
		}
		const std::vector<float> q = query();
		const int best = db->search_mode(Jump, q.data());
		if (best < 0) {
			return false;
		}
		const int bc = db->pose_clip[size_t(best)];
		const int bf = db->pose_frame[size_t(best)];
		const Clip &c = db->clips[size_t(bc)];
		clip = bc;
		time = float(bf) / c.fps;
		locked_until = float(c.lands[size_t(bf)]) / c.fps + 0.2f;
		last_cost = db->cost(best, q.data());
		since_search = 0;
		++switches;
		++jumps;
		return true;
	}

	bool search() {
		const std::vector<float> q = query();
		int best = db->search_mode(mode, q.data());
		++searches;
		if (best < 0) {
			return false;
		}
		const Clip &c = db->clips[size_t(clip)];
		const int now_f = std::clamp(int(std::lround(time * c.fps)), 0, c.frames - 1);
		int now_pose = 0;
		for (int i = 0; i < clip; ++i) {
			now_pose += db->clips[size_t(i)].frames;
		}
		now_pose += now_f;
		const float now_cost = db->cost(now_pose, q.data());
		const bool at_end = !c.loop && time >= float(c.frames - 1) / c.fps - 1e-4f;
		// A pose of another kind than asked (standing while crouch is asked) always gives way.
		const int want = db->has_mode(mode) ? mode : int(Stand);
		const bool wrong_kind = c.kind[size_t(now_f)] != want;
		int bc = db->pose_clip[size_t(best)];
		float bt = float(db->pose_frame[size_t(best)]) / db->clips[size_t(bc)].fps;
		// A gait cycle is cyclic: going back in time, or to another clip, is a join only at the
		// same phase. Out of phase, play on, or, when the clip must change, take the best pose in
		// phase.
		const float now_phase = c.phase[size_t(now_f)];
		const float tol = db->settings.phase_tol;
		const int now_curve = c.curve[size_t(now_f)];
		auto out_of_phase = [&](int ci, float t) {
			const Clip &k = db->clips[size_t(ci)];
			const int f = std::clamp(int(std::lround(t * k.fps)), 0, k.frames - 1);
			// Learned phases line up only on one manifold curve; foot-strike phases (curve -1) always do.
			if (k.curve[size_t(f)] != now_curve) {
				return false;
			}
			return tol > 0 && Database::phase_gap(now_phase, k.phase[size_t(f)]) > tol;
		};
		if (out_of_phase(bc, bt)) {
			if (!at_end && !wrong_kind) {
				return false;
			}
			best = db->search_mode_except(mode, q.data(), out_of_phase);
			if (best < 0) {
				return false;
			}
			bc = db->pose_clip[size_t(best)];
			bt = float(db->pose_frame[size_t(best)]) / db->clips[size_t(bc)].fps;
		}
		const float best_cost = db->cost(best, q.data());
		last_cost = best_cost;
		const bool near_now = bc == clip && std::fabs(bt - time) < db->settings.same_clip_window;
		if (near_now && !at_end && !wrong_kind) {
			return false;
		}
		if (!at_end && !wrong_kind && best_cost > now_cost - db->settings.switch_margin) {
			return false;
		}
		clip = bc;
		time = bt;
		++switches;
		return true;
	}

	v3 desired_cache;
	float facing_cache = std::numeric_limits<float>::quiet_NaN();
	bool update(float dt, v3 desired, float facing_yaw, int want_mode = Stand, bool want_jump = false) {
		desired_cache = desired;
		facing_cache = facing_yaw;
		if (want_mode != mode && want_mode >= Stand && want_mode <= Crawl) {
			mode = want_mode;
			since_search = 1e9f; // answer a new mode at once
		}
		bool switched = false;
		if (want_jump && mode == Stand) {
			switched = jump();
		}
		return step(dt, desired, facing_yaw) || switched;
	}
};

} // namespace mm
