// A steady gait as a loop (RFD66606-2327, "Steady gaits loop through a seam"): cut a clip in
// motion.elf's layout to whole strides, fade its end into its start carried by one loop's travel,
// and measure the seam by joint angle against the clip's ordinary frame steps.
#pragma once
#include "mm_core.hpp"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace mm {

using q4 = std::array<float, 4>; // w, x, y, z

inline q4 quat_of(const m3 &r) {
	const float *m = r.m;
	const float tr = m[0] + m[4] + m[8];
	q4 q;
	if (tr > 0) {
		const float s = std::sqrt(tr + 1.0f) * 2;
		q = {0.25f * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s};
	} else if (m[0] > m[4] && m[0] > m[8]) {
		const float s = std::sqrt(1.0f + m[0] - m[4] - m[8]) * 2;
		q = {(m[7] - m[5]) / s, 0.25f * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s};
	} else if (m[4] > m[8]) {
		const float s = std::sqrt(1.0f + m[4] - m[0] - m[8]) * 2;
		q = {(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25f * s, (m[5] + m[7]) / s};
	} else {
		const float s = std::sqrt(1.0f + m[8] - m[0] - m[4]) * 2;
		q = {(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25f * s};
	}
	const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

inline m3 mat_of(const q4 &q) {
	const float w = q[0], x = q[1], y = q[2], z = q[3];
	m3 r;
	r.m[0] = 1 - 2 * (y * y + z * z);
	r.m[1] = 2 * (x * y - w * z);
	r.m[2] = 2 * (x * z + w * y);
	r.m[3] = 2 * (x * y + w * z);
	r.m[4] = 1 - 2 * (x * x + z * z);
	r.m[5] = 2 * (y * z - w * x);
	r.m[6] = 2 * (x * z - w * y);
	r.m[7] = 2 * (y * z + w * x);
	r.m[8] = 1 - 2 * (x * x + y * y);
	return r;
}

inline q4 slerp(q4 a, q4 b, float t) {
	float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
	if (d < 0) {
		b = {-b[0], -b[1], -b[2], -b[3]};
		d = -d;
	}
	float wa = 1 - t, wb = t;
	if (d < 0.9995f) {
		const float th = std::acos(d);
		wa = std::sin((1 - t) * th) / std::sin(th);
		wb = std::sin(t * th) / std::sin(th);
	}
	q4 r = {a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb, a[2] * wa + b[2] * wb, a[3] * wa + b[3] * wb};
	const float n = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
	return {r[0] / n, r[1] / n, r[2] / n, r[3] / n};
}

// Degrees between two rotations.
inline float angle_deg(const m3 &a, const m3 &b) {
	const q4 p = quat_of(a), q = quat_of(b);
	const float d = std::min(1.0f, std::fabs(p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3]));
	return 2 * std::acos(d) * 180.0f / kPi;
}

inline m3 rot_y(float yaw) {
	const float c = std::cos(yaw), s = std::sin(yaw);
	m3 r;
	r.m[0] = c;
	r.m[2] = s;
	r.m[6] = -s;
	r.m[8] = c;
	return r;
}

// A clip as frames x joints of (parent-local rotation, translation).
struct Frames {
	size_t J = 0;
	int F = 0;
	std::vector<m3> rot;
	std::vector<v3> pos;
	int root = 0; // the joint without a parent: it carries the clip's travel
	m3 &r(int f, size_t j) { return rot[size_t(f) * J + j]; }
	v3 &p(int f, size_t j) { return pos[size_t(f) * J + j]; }
	const m3 &r(int f, size_t j) const { return rot[size_t(f) * J + j]; }
	const v3 &p(int f, size_t j) const { return pos[size_t(f) * J + j]; }
};

inline Frames frames_of(const std::vector<float> &xf, const std::vector<int32_t> &parents) {
	Frames c;
	c.J = parents.size();
	c.F = int(xf.size() / (c.J * 12));
	for (size_t j = 0; j < c.J; ++j) {
		if (parents[j] < 0) {
			c.root = int(j);
		}
	}
	for (int f = 0; f < c.F; ++f) {
		for (size_t j = 0; j < c.J; ++j) {
			const float *o = &xf[(size_t(f) * c.J + j) * 12];
			m3 r;
			std::copy(o, o + 9, r.m);
			c.rot.push_back(r);
			c.pos.push_back({o[9], o[10], o[11]});
		}
	}
	return c;
}

inline std::vector<float> floats_of(const Frames &c) {
	std::vector<float> xf;
	for (int f = 0; f < c.F; ++f) {
		for (size_t j = 0; j < c.J; ++j) {
			const m3 &r = c.r(f, j);
			xf.insert(xf.end(), r.m, r.m + 9);
			const v3 &t = c.p(f, j);
			xf.insert(xf.end(), {t.x, t.y, t.z});
		}
	}
	return xf;
}

// The root's facing at a frame: its forward (+Z) on the floor, averaged over `half` frames either
// side as the database does.
inline float facing(const Frames &c, int f, int half) {
	v3 s;
	for (int k = std::max(0, f - half); k <= std::min(c.F - 1, f + half); ++k) {
		const v3 fw = mul(c.r(k, size_t(c.root)), v3{0, 0, 1});
		s = s + v3{fw.x, 0, fw.z};
	}
	return std::atan2(s.x, s.z);
}

// The floor move taking frame a's root onto frame b's: a turn about Y and a shift on the floor.
struct Carry {
	float yaw = 0;
	v3 from, to;
	v3 point(v3 p) const {
		const v3 d = rotate_y(v3{p.x - from.x, 0, p.z - from.z}, yaw);
		return {to.x + d.x, p.y, to.z + d.z};
	}
	m3 turn(const m3 &r) const { return mul(rot_y(yaw), r); }
};

inline Carry carry(const Frames &c, int a, int b, int half) {
	Carry t;
	t.yaw = wrap_angle(facing(c, b, half) - facing(c, a, half));
	t.from = c.p(a, size_t(c.root));
	t.to = c.p(b, size_t(c.root));
	return t;
}

// The largest joint turn (degrees) from frame a to frame b, b's root first moved by `t` when
// given: the root by its world rotation, the other joints by their local ones.
inline float step_deg(const Frames &c, int a, const Frames &d, int b, const Carry *t = nullptr) {
	float worst = 0;
	for (size_t j = 0; j < c.J; ++j) {
		const m3 rb = (t && int(j) == c.root) ? t->turn(d.r(b, j)) : d.r(b, j);
		worst = std::max(worst, angle_deg(c.r(a, j), rb));
	}
	return worst;
}

struct LoopCut {
	int from = 0, to = 0;    // source frames: the loop is [from, to], `to` the next pass's `from`
	float seam_deg = 0;      // the largest joint step over the fade and the wrap
	float ordinary_deg = 0;  // the 95th percentile of the source clip's frame steps
	float cost = 0;
	Frames out;
	bool ok() const { return seam_deg <= ordinary_deg; }
};

// The ordinary frame step of a clip: the 95th percentile of the largest joint turn per step.
inline float ordinary_step(const Frames &c) {
	std::vector<float> s;
	for (int f = 0; f + 1 < c.F; ++f) {
		s.push_back(step_deg(c, f, c, f + 1));
	}
	std::sort(s.begin(), s.end());
	return s.empty() ? 0.0f : s[s.size() * 95 / 100];
}

// The seam of a clip played as a loop: the largest joint step over its last `fade` frames and
// across the wrap (last frame to the second, the second moved by one loop's travel).
inline float seam_step(const Frames &c, int fade) {
	float worst = 0;
	for (int f = std::max(0, c.F - 1 - fade); f + 1 < c.F; ++f) {
		worst = std::max(worst, step_deg(c, f, c, f + 1));
	}
	const Carry t = carry(c, 0, c.F - 1, 0);
	return std::max(worst, step_deg(c, c.F - 1, c, 1, &t));
}

// Keypoint features per frame: every joint's position in the root's floor frame, and its speed,
// each dimension scaled to unit spread over the clip.
inline std::vector<std::vector<float>> keypoint_features(const Frames &c, const std::vector<int32_t> &parents, float fps,
		int half) {
	std::vector<std::vector<v3>> world(size_t(c.F), std::vector<v3>(c.J));
	for (int f = 0; f < c.F; ++f) {
		std::vector<m3> gr(c.J);
		for (size_t j = 0; j < c.J; ++j) {
			const int p = parents[j];
			if (p < 0) {
				gr[j] = c.r(f, j);
				world[size_t(f)][j] = c.p(f, j);
			} else {
				gr[j] = mul(gr[size_t(p)], c.r(f, j));
				world[size_t(f)][j] = world[size_t(f)][size_t(p)] + mul(gr[size_t(p)], c.p(f, j));
			}
		}
	}
	std::vector<std::vector<float>> feat(size_t(c.F));
	for (int f = 0; f < c.F; ++f) {
		const v3 o = c.p(f, size_t(c.root));
		const float yaw = facing(c, f, half);
		const int g = std::min(f + 1, c.F - 1), h = std::max(g - 1, 0);
		for (size_t j = 0; j < c.J; ++j) {
			const v3 l = to_local(world[size_t(f)][j], v3{o.x, 0, o.z}, yaw);
			const v3 v = rotate_y((world[size_t(g)][j] - world[size_t(h)][j]) * fps, -yaw);
			feat[size_t(f)].insert(feat[size_t(f)].end(), {l.x, l.y, l.z, v.x, v.y, v.z});
		}
	}
	const size_t D = feat.empty() ? 0 : feat[0].size();
	for (size_t d = 0; d < D; ++d) {
		float m = 0, s = 0;
		for (const std::vector<float> &row : feat) {
			m += row[d];
		}
		m /= float(c.F);
		for (const std::vector<float> &row : feat) {
			s += (row[d] - m) * (row[d] - m);
		}
		s = std::sqrt(s / float(c.F)) + 1e-4f;
		for (std::vector<float> &row : feat) {
			row[d] = (row[d] - m) / s;
		}
	}
	return feat;
}

// Cut [from, to] with to - from at least half the clip, then fade the last `fade` frames into the
// frames before `from`, carried by one loop's travel, so the last frame is the first frame moved
// on by one loop. The pairs whose keypoints agree best are tried; the smallest seam wins.
inline Frames faded(const Frames &c, int from, int to, int fade) {
	const Carry t = carry(c, from, to, 0);
	Frames o;
	o.J = c.J;
	o.root = c.root;
	o.F = to - from + 1;
	for (int f = from; f <= to; ++f) {
		const int k = f - (to - fade); // fade step, 0..fade
		const int g = from - (to - f); // the source frame this one fades into
		const float w = k <= 0 ? 0.0f : float(k) / float(fade);
		const float u = w * w * (3 - 2 * w);
		for (size_t j = 0; j < c.J; ++j) {
			m3 r = c.r(f, j);
			v3 p = c.p(f, j);
			if (u > 0) {
				const bool root = int(j) == c.root;
				const m3 rt = root ? t.turn(c.r(g, j)) : c.r(g, j);
				const v3 pt = root ? t.point(c.p(g, j)) : c.p(g, j);
				r = mat_of(slerp(quat_of(r), quat_of(rt), u));
				p = p * (1 - u) + pt * u;
			}
			o.rot.push_back(r);
			o.pos.push_back(p);
		}
	}
	return o;
}

// The loop runs from one left-foot strike to a later one at least two gait cycles on, every frame
// between inside a cycle (`phase` >= 0, from Database::cycle_phase), so it is whole strides of
// the clip's gait and never a stretch of standing. Of those runs, the smallest seam wins.
inline LoopCut cut_loop(const std::vector<float> &xf, const std::vector<int32_t> &parents, float fps,
		const std::vector<float> &phase, float fade_s = 0.2f) {
	LoopCut out;
	const Frames c = frames_of(xf, parents);
	const int fade = std::max(1, int(std::lround(fade_s * fps)));
	std::vector<int> strikes;
	for (int f = 0; f < c.F; ++f) {
		if (phase[size_t(f)] == 0.0f) {
			strikes.push_back(f);
		}
	}
	out.ordinary_deg = ordinary_step(c);
	out.seam_deg = std::numeric_limits<float>::max();
	for (size_t i = 0; i < strikes.size(); ++i) {
		for (size_t j = i + 2; j < strikes.size(); ++j) {
			const int a = strikes[i], b = strikes[j];
			bool cycling = a >= fade;
			for (int f = a; f < b && cycling; ++f) {
				cycling = phase[size_t(f)] >= 0;
			}
			if (!cycling) {
				break;
			}
			Frames o = faded(c, a, b, fade);
			const float seam = seam_step(o, fade);
			if (seam < out.seam_deg - 1e-4f || (seam < out.seam_deg + 1e-4f && b - a > out.to - out.from)) {
				out.seam_deg = seam;
				out.from = a;
				out.to = b;
				out.out = std::move(o);
			}
		}
	}
	return out;
}

} // namespace mm
