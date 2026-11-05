"""Run model-written code against tests in a separate Python process.

Isolation (Windows, no Docker):
  * a fresh interpreter (`python -I`, no site/user paths from the environment), run in a temp directory;
  * a wall-clock timeout (the process is killed when it expires);
  * a Windows Job Object that caps committed memory per process, limits the number of child processes to 1,
    and kills everything in the job when the job handle closes.

Not isolated: network access and filesystem reads outside the temp directory. The harness is for measuring
reward hacking on a local machine, not for running untrusted code from the internet.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

RUNNER = r'''
import contextlib, io, json, sys
src = json.load(open(sys.argv[1], encoding="utf-8"))
out = {"setup_ok": False, "solution_ok": False, "tests": [], "error": ""}
ns = {}
try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src["setup"], ns)
        out["setup_ok"] = True
        exec(compile(src["code"], "<solution>", "exec"), ns)
        out["solution_ok"] = True
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"[:300]
for t in src["tests"]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(t, ns)
        out["tests"].append(True)
    except BaseException:
        out["tests"].append(False)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f)
'''

def _windows_job():
    """Create a Job Object with memory and process limits. Returns the handle, or None off Windows."""
    global _JOB_OK
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]

    class EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x8
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    info = EXT()
    info.BasicLimitInformation.LimitFlags = (JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_PROCESS_MEMORY
                                             | JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = 512 * 1024 * 1024
    info.JobMemoryLimit = 512 * 1024 * 1024
    JobObjectExtendedLimitInformation = 9
    ok = kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
    if not ok:
        kernel32.CloseHandle(job)
        return None
    return job


def _assign(job, proc):
    if job is None:
        return
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = ctypes.c_void_p(int(proc._handle))
    kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), handle)


@dataclass
class ExecResult:
    setup_ok: bool = False
    solution_ok: bool = False
    tests: list = field(default_factory=list)
    error: str = ""
    timed_out: bool = False

    def all_pass(self, idx):
        return self.solution_ok and bool(idx) and all(self.tests[i] for i in idx)


def run_tests(code, setup, tests, timeout_s=6.0):
    """Run `code` (the solution) then each test string in `tests`, in a sandboxed child process.

    Returns an ExecResult. `tests[i]` is True if that assert ran without raising.
    """
    with tempfile.TemporaryDirectory(prefix="rhack_") as tmp:
        src_path = os.path.join(tmp, "src.json")
        out_path = os.path.join(tmp, "out.json")
        runner_path = os.path.join(tmp, "runner.py")
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump({"setup": setup or "", "code": code, "tests": list(tests)}, f)
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(RUNNER)
        # venv pythons on Windows are launchers that start a second process; use the real interpreter
        # so the job's one-process limit holds.
        interpreter = getattr(sys, "_base_executable", sys.executable)
        proc = subprocess.Popen(
            [interpreter, "-I", runner_path, src_path, out_path],
            cwd=tmp, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=(0x08000000 if os.name == "nt" else 0),  # CREATE_NO_WINDOW
        )
        job = _windows_job()
        _assign(job, proc)
        try:
            proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True
        finally:
            if job is not None:
                import ctypes
                ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(job))
        if timed_out or not os.path.exists(out_path):
            return ExecResult(error="timeout" if timed_out else "no output", timed_out=timed_out,
                              tests=[False] * len(tests))
        with open(out_path, encoding="utf-8") as f:
            d = json.load(f)
        return ExecResult(setup_ok=d["setup_ok"], solution_ok=d["solution_ok"], tests=d["tests"], error=d["error"])


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(completion):
    """Take the first fenced code block if present, else the whole completion."""
    m = _FENCE.search(completion)
    return m.group(1) if m else completion
