# Offline verification

158 tests passed; no online mission was run.

| Test module | Tests | Result |
| --- | ---: | --- |
| test_room_search_v1.py | 43 | PASS |
| test_room_search_zero_observation_feedback.py | 7 | PASS |
| test_room_search_grid_status_pair_repair.py | 20 | PASS |
| test_room_search_completion_contract.py | 13 | PASS |
| test_room_local_online_enablement.py | 8 | PASS |
| test_portal_pthrough_profile_r41.py | 3 | PASS |
| test_room_search_task_directed.py | 44 | PASS |
| test_room_search_candidate_evidence_instrumentation.py | 6 | PASS |
| test_room_search_revisit_efficiency.py | 7 | PASS |
| test_room_search_stage1_reduction.py | 7 | PASS |

Command pattern (from repository root):

```bash
PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages LD_LIBRARY_PATH=/opt/ros/noetic/lib PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/local_subgoal_runner_mvp/tests -p test_room_search_v1.py
```

Replace the test filename with each module above. The selected tests cover ranking/admission, observation feedback, grid/status pairing, completion, portal budgets and room-local enablement.

The P_through test was restored to its historical 16-step expectation. One old completion fixture lacked path_length_m; it was given 1.0 to exercise valid zero-gain completion. Core production hashes remain unchanged.
