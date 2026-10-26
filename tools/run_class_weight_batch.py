"""Persistent seed42 formal queue: cache, four runs, val export, paired report.

Stops on any failed subprocess. Paused sessions resume their own last.pt;
failed runs are never silently retried or replaced by another attempt.
"""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.train_expanded import ROOT, atomic_json, sha256
from tools.train_class_weight import code_fingerprint
from tools.run_expanded_session import system_memory


RUNS = (
    ("a_b1", "a", "post", "baseline", "b1"),
    ("a_b2", "a", "early", "baseline", "b2"),
    ("b_b1", "b", "post", "class_weight", "b1"),
    ("b_b2", "b", "early", "class_weight", "b2"),
)
REPORT = ROOT / "reports/within_event_mm/class_weight"


def now():
    return datetime.datetime.now().astimezone().isoformat()


def run_path(scheme, model):
    return ROOT / "output/within_event_mm" / scheme / model / "seed42/attempt01"


class Queue:
    def __init__(self, directory):
        self.directory = directory
        self.sources = {}
        self.state = {"status": "preparing", "runner_pid": os.getpid(), "started": now(),
                      "phase": "source_freeze", "runs": {r[0]: "queued" for r in RUNS},
                      "seed": 42, "attempt": "attempt01", "test_evaluated": False}

    def save(self, **changes):
        self.state.update(changes, updated=now())
        atomic_json(self.directory / "status.json", self.state)
        text = ["# Formal seed42 queue", "", f"Updated: {self.state['updated']}",
                f"Status: {self.state['status']}; phase: {self.state['phase']}", "",
                "| Run | Status |", "| --- | --- |"]
        text += [f"| {k} | {v} |" for k, v in self.state["runs"].items()]
        text += ["", "Test remains sealed. Scores are not final until all four runs and paired validation finish.",
                 "", f"Local queue evidence: `{self.directory.relative_to(ROOT).as_posix()}`", ""]
        temporary = REPORT / "RUN_STATUS.md.partial"
        temporary.write_text("\n".join(text), encoding="utf-8")
        temporary.replace(REPORT / "RUN_STATUS.md")

    def freeze(self):
        accepted = json.loads((REPORT / "verification.json").read_text(encoding="utf-8"))
        self.sources = dict(accepted["source_sha256"])
        for name, digest in self.sources.items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Source differs from accepted implementation: {name}")
        self.sources.update(code_fingerprint())
        self.sources["tools/run_class_weight_batch.py"] = sha256(Path(__file__))
        for name, digest in self.sources.items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Source differs from accepted implementation: {name}")
            target = self.directory / "frozen_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        atomic_json(self.directory / "source_lock.json", self.sources)

    def check_sources(self):
        for name, digest in self.sources.items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Frozen source changed during formal queue: {name}")

    def execute(self, phase, arguments, run=None):
        self.check_sources()
        self.save(status="running", phase=phase)
        command = [sys.executable, "-u", "-m", *arguments]
        stamp = str(time.time_ns())
        with (self.directory / f"{phase}_{stamp}.stdout.log").open("w", encoding="utf-8") as stdout, \
                (self.directory / f"{phase}_{stamp}.stderr.log").open("w", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            self.save(child_pid=child.pid, command=command)
            while child.poll() is None:
                sample = {"time": now(), "phase": phase, **system_memory()}
                if run and (run / "status.json").exists():
                    sample["training"] = json.loads((run / "status.json").read_text(encoding="utf-8"))
                try:
                    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                    sample["gpu_utilization_memory_mb_temperature"] = gpu.stdout.strip()
                except (OSError, subprocess.TimeoutExpired) as exc:
                    sample["gpu_error"] = repr(exc)
                atomic_json(self.directory / "resources_latest.json", sample)
                with (self.directory / "resources.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample) + "\n")
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode:
                raise RuntimeError(f"{phase} failed (exit {child.returncode}); inspect {stderr.name}")
        self.check_sources()

    def collect(self, scheme, model, run):
        target = ROOT / "reports/within_event_mm" / scheme / model / "seed42/attempt01"
        target.mkdir(parents=True, exist_ok=False)
        for name in ("summary.json", "history.jsonl", "run_metadata.json"):
            shutil.copyfile(run / name, target / name)
        shutil.copyfile(run / "validation/evaluation.json", target / "evaluation.json")

    def finish_report(self):
        comparison = ROOT / "output/within_event_mm/class_weight/comparison_seed42_v1"
        values = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
        if set(values["statuses"].values()) != {"complete"}:
            raise ValueError("All four runs must be complete")
        shutil.copyfile(comparison / "results.csv", REPORT / "results.csv")
        shutil.copyfile(comparison / "comparison.json", REPORT / "comparison.json")
        text = ["# Class-weight comparison: seed42", "", "All four runs completed; validation only. Test remains sealed.",
                "Single-seed exploratory evidence; no stability or significance claim.", "",
                "| Model | A val Macro F1 | B val Macro F1 | B minus A |", "| --- | ---: | ---: | ---: |"]
        for model, pair in values["pairs"].items():
            a = pair["a"]["pooled"]["metrics"]["macro_f1"]
            b = pair["b"]["pooled"]["metrics"]["macro_f1"]
            text.append(f"| {model} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
        text += ["", "## Prespecified error checks", "",
                 "Counts, true-class supports, rates, all class metrics and confusion matrices are in [comparison.json](comparison.json).",
                 "Corrected and newly introduced errors are both retained; cases use the prespecified ID-hash grouping rule.", "",
                 "| Model | Event | Check | A count/support | B count/support | Rate change |",
                 "| --- | --- | --- | ---: | ---: | ---: |"]
        for model, pair in values["pairs"].items():
            for event in pair["a"]:
                for name, a in pair["a"][event]["checks"].items():
                    b = pair["b"][event]["checks"][name]
                    delta = f"{b['rate']-a['rate']:+.6f}" if a["rate"] is not None and b["rate"] is not None else "NA"
                    text.append(f"| {model} | {event} | {name} | {a['count']}/{a['support']} | {b['count']}/{b['support']} | {delta} |")
        text += ["", "A is rebuilt under the same accelerated pipeline; historical results are reference only.",
                 "These are best-on-validation development scores, not independent test scores.",
                 "Full checkpoints, predictions, cache and logs remain in local output; no Release or online publication has occurred.", ""]
        (REPORT / "analysis_report.md").write_text("\n".join(text), encoding="utf-8")

    def run(self):
        self.freeze()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if shutil.disk_usage(ROOT).free < 20 * 1024**3:
            raise RuntimeError("Less than 20 GiB free space")
        for _, _, _, scheme, model in RUNS:
            path = run_path(scheme, model)
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"Formal attempt already exists: {path}")
        config_a = "configs/within_event_mm/class_weight/a_seed42.json"
        config = json.loads((ROOT / config_a).read_text())
        cache = ROOT / config["cache_dir"]
        if cache.exists():
            from stage_b_damage.within_event_cache import verify_cache
            verify_cache(config, ROOT)
        else:
            self.execute("cache", ["tools.build_within_event_cache", "--config", config_a])
        for key, condition, mode, scheme, model in RUNS:
            path = run_path(scheme, model)
            self.state["runs"][key] = "running"
            args = ["tools.train_class_weight", "--config", f"configs/within_event_mm/class_weight/{condition}_seed42.json",
                    "--mode", mode, "--output-dir", str(path), "--session-hours", "8"]
            for session in range(1, 22):
                self.execute(f"{key}_session{session}", args + (["--resume"] if session > 1 else []), path)
                summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
                if summary["status"] == "complete":
                    break
                if summary["status"] != "paused":
                    raise RuntimeError(f"Unexpected run status: {summary['status']}")
            else:
                raise RuntimeError("Session limit reached")
            self.execute(f"{key}_val", ["evaluation.class_weight", "export", "--run", str(path)])
            self.collect(scheme, model, path)
            self.state["runs"][key] = "complete"
            self.save()
        args = ["evaluation.class_weight", "compare"]
        for key, _, _, scheme, model in RUNS:
            args += ["--" + key.replace("_", "-"), str(run_path(scheme, model))]
        args += ["--output", str(ROOT / "output/within_event_mm/class_weight/comparison_seed42_v1")]
        self.execute("comparison", args)
        self.finish_report()
        self.save(status="complete", phase="complete", finished=now(), child_pid=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.queue_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    queue = Queue(directory)
    awake = False
    try:
        if os.name == "nt":
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            if not awake:
                raise RuntimeError("Could not inhibit automatic sleep")
        queue.save(sleep_inhibited=awake)
        queue.run()
    except BaseException as exc:
        queue.state["runs"] = {k: "failed" if v == "running" else v for k, v in queue.state["runs"].items()}
        queue.save(status="failed", error=repr(exc), finished=now())
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
