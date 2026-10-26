# B2 structure formal queue

{
  "status": "complete",
  "runner_pid": 3648,
  "started": "2026-10-26T05:47:03.958442+08:00",
  "phase": "complete",
  "runs": {
    "a_b2": "complete",
    "b_b2": "complete"
  },
  "seed": 42,
  "attempt": "attempt01",
  "test_evaluated": false,
  "experiment": "b2_structure",
  "deadline": "2026-10-26T07:50:00+08:00",
  "sleep_inhibited": true,
  "updated": "2026-10-26T06:48:42.379944+08:00",
  "child_pid": null,
  "command": [
    "D:\\DDA4220\\DDA4220\\.venv\\Scripts\\python.exe",
    "-u",
    "-m",
    "evaluation.b2_structure",
    "compare",
    "--a-b2",
    "D:\\DDA4220\\DDA4220\\output\\within_event_mm\\b2_structure\\a\\b2\\seed42\\attempt01",
    "--b-b2",
    "D:\\DDA4220\\DDA4220\\output\\within_event_mm\\b2_structure\\b\\b2\\seed42\\attempt01",
    "--output",
    "D:\\DDA4220\\DDA4220\\output\\within_event_mm\\b2_structure\\session_seed42_attempt01\\comparison"
  ],
  "finished": "2026-10-26T06:48:42.379944+08:00"
}
