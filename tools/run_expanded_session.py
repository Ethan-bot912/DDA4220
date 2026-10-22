"""Persistent single-model session with resource telemetry and sleep inhibition.

Run from repository root. Launch with Start-Process to outlive the terminal.
Child trainer owns epoch-boundary stopping. Does not change Windows power plans.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def system_memory():
    if os.name != "nt":
        return {}
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    m = Memory()
    m.length = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return {"error": "GlobalMemoryStatusEx failed"}
    return {"ram_total_bytes": m.total, "ram_available_bytes": m.available, "ram_load_percent": m.load}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["post", "early"], required=True)
    p.add_argument("--config", default="configs/expanded_seed42.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--session-hours", type=float, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-epoch", type=int, default=0)
    a = p.parse_args()
    os.chdir(ROOT)
    out = a.output_dir.resolve()
    config_path = Path(a.config)
    audit_path = ROOT / json.loads(config_path.read_text(encoding="utf-8-sig"))["manifest"]
    audit = json.loads((audit_path.parent / "geotiff_audit.json").read_text())
    if audit["errors"]:
        raise ValueError("Image audit failed")
    if not (audit_path.parent / "data_lock.json").exists():
        raise FileNotFoundError("Prepare data_lock.json before formal training")
    data_lock = json.loads((audit_path.parent / "data_lock.json").read_text())
    if hashlib.sha256(audit_path.read_bytes()).hexdigest() != data_lock["manifest_sha256"]:
        raise ValueError("Manifest differs from prepared data lock")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != data_lock["config_sha256"]:
        raise ValueError("Configuration differs from prepared data lock")
    for index, record in enumerate(data_lock["files"], 1):
        path = ROOT / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing/size-mismatched data file: {path}")
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != record["sha256"]:
            raise ValueError(f"Data bytes differ from lock: {path}")
        if index % 1000 == 0:
            print(f"Verified {index}/{len(data_lock['files'])} data files", flush=True)
    if out.exists() and not a.resume and any(out.iterdir()):
        raise FileExistsError("Output exists; resume explicitly or choose a new output directory")
    # Trainer creates output; telemetry stays outside it so its nonempty guard works.
    monitoring = out.parent / "monitoring"
    monitoring.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = monitoring / f"{a.mode}_{stamp}"
    command = [sys.executable, "-u", "-m", "tools.train_expanded", "--config", a.config,
               "--mode", a.mode, "--output-dir", str(out), "--session-hours", str(a.session_hours)]
    if a.resume:
        command.append("--resume")
    if a.stop_after_epoch:
        command.extend(["--stop-after-epoch", str(a.stop_after_epoch)])
    keep_awake = False
    try:
        if os.name == "nt":
            keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            if not keep_awake:
                raise RuntimeError("Could not inhibit automatic sleep")
        with prefix.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, prefix.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            info = {"runner_pid": os.getpid(), "trainer_pid": process.pid, "command": command,
                    "started_local": datetime.datetime.now().astimezone().isoformat(),
                    "session_hours": a.session_hours, "sleep_inhibited": keep_awake,
                    "stdout": str(prefix.with_suffix(".stdout.log")), "stderr": str(prefix.with_suffix(".stderr.log"))}
            (monitoring / f"{a.mode}_session.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            while True:
                sample = {"time_local": datetime.datetime.now().astimezone().isoformat(), **system_memory()}
                try:
                    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=15)
                    sample["gpu_csv"] = gpu.stdout.strip()
                except (OSError, subprocess.TimeoutExpired) as exc:
                    sample["gpu_error"] = repr(exc)
                try:
                    sample["training"] = json.loads((out / "status.json").read_text())
                except (OSError, ValueError):
                    pass
                with prefix.with_suffix(".resources.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(sample) + "\n")
                (monitoring / f"{a.mode}_resources_latest.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
                result = process.poll()
                if result is not None:
                    info.update(returncode=result, finished_local=datetime.datetime.now().astimezone().isoformat())
                    (monitoring / f"{a.mode}_session.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
                    return result
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    raise SystemExit(main())
