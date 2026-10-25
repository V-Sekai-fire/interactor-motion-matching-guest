# Bridge joins with crawl_idle_a in the database

- Same 7 bridge clips, same metric and controls as before (tools/bridges/body.gd --joins, tools/bridges/joins.exs); the only change is crawl_idle_a (kind 2, derived by mm_kinds) among the database clips.
- PASS 14/14: neighbours, planted and self controls hold on every join.
- The three ends that sat on a still hands-and-knees pose are unchanged: crawl_to_stand@0 290.5 mm (crawl_down_up@164), crouch_to_crawl@299 288.4 mm (crawl_down_up@163), stand_to_crawl@299 297.7 mm (crawl_down_up@166) mean.
- crawl_idle_a is never the pick; its best frame against those ends is 300.0 mm (frame 167), 298.1 mm (frame 168) and 318.0 mm (frame 167) mean, 497 to 499 mm max.
- So crawl_idle_a does not give those ends a close database frame.

```
JOIN into crawl_to_crouch (crawl_to_crouch@0, kind 2) <-> crawl_slow@24: max 69.8 mean 30.8 mm; one frame off @23 (kind 2) mean 32.8 max 71.4, @25 (kind 2) mean 33.0 max 73.3
JOIN out of crawl_to_crouch (crawl_to_crouch@299, kind 2) <-> crawl_stop@299: max 49.1 mean 17.9 mm; one frame off @298 (kind 2) mean 18.0 max 48.8
JOIN into crawl_to_stand (crawl_to_stand@0, kind 2) <-> crawl_down_up@164: max 515.5 mean 290.5 mm; one frame off @163 (kind 2) mean 291.2 max 517.4, @165 (kind 2) mean 292.1 max 514.0
JOIN out of crawl_to_stand (crawl_to_stand@299, kind 0) <-> run@136: max 89.0 mean 20.5 mm; one frame off @135 (kind 0) mean 20.6 max 89.3, @137 (kind 0) mean 20.5 max 88.6
JOIN into crouch_to_crawl (crouch_to_crawl@0, kind 2) <-> crawl_right@39: max 193.4 mean 66.9 mm; one frame off @38 (kind 2) mean 68.0 max 199.6, @40 (kind 2) mean 67.3 max 186.6
JOIN out of crouch_to_crawl (crouch_to_crawl@299, kind 2) <-> crawl_down_up@163: max 504.0 mean 288.4 mm; one frame off @162 (kind 2) mean 290.8 max 505.7, @164 (kind 2) mean 289.4 max 502.3
JOIN into crouch_to_stand (crouch_to_stand@0, kind 1) <-> crouch_down_up@143: max 220.8 mean 77.8 mm; one frame off @142 (kind 1) mean 77.8 max 220.8, @144 (kind 1) mean 77.8 max 220.9
JOIN out of crouch_to_stand (crouch_to_stand@299, kind 0) <-> turn_right@115: max 120.7 mean 24.2 mm; one frame off @114 (kind 0) mean 24.6 max 120.9, @116 (kind 0) mean 24.6 max 120.6
JOIN into run_to_stand (run_to_stand@0, kind 0) <-> run_long@0: max 106.4 mean 33.8 mm; one frame off @1 (kind 0) mean 41.3 max 117.7
JOIN out of run_to_stand (run_to_stand@299, kind 0) <-> run_long@250: max 52.7 mean 20.6 mm; one frame off @249 (kind 0) mean 20.6 max 52.0, @251 (kind 0) mean 20.6 max 53.4
JOIN into stand_to_crawl (stand_to_crawl@0, kind 0) <-> sprint@189: max 60.8 mean 18.5 mm; one frame off @188 (kind 0) mean 18.5 max 61.0, @190 (kind 0) mean 18.5 max 60.7
JOIN out of stand_to_crawl (stand_to_crawl@299, kind 2) <-> crawl_down_up@166: max 515.2 mean 297.7 mm; one frame off @165 (kind 2) mean 298.3 max 496.9, @167 (kind 2) mean 298.9 max 537.0
JOIN into walk_to_stand (walk_to_stand@0, kind 0) <-> walk_jump@36: max 70.3 mean 28.7 mm; one frame off @35 (kind 0) mean 29.8 max 95.8, @37 (kind 0) mean 33.3 max 107.8
JOIN out of walk_to_stand (walk_to_stand@299, kind 0) <-> stop_walk@285: max 22.3 mean 9.3 mm; one frame off @284 (kind 0) mean 9.3 max 23.1, @286 (kind 0) mean 9.7 max 21.7
PASS 14/14
```
