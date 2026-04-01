from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = ROOT / "code"
DEPS_ROOT = ROOT / ".pydeps"
OUTPUT_ROOT = ROOT / "outputs"


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(CODE_ROOT)]
    if DEPS_ROOT.exists():
        python_paths.append(str(DEPS_ROOT))
    pythonpath = os.pathsep.join(python_paths)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    if extra:
        env.update(extra)
    return env


def run_module(module: str, default_args: list[str] | None = None) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", module]
    if default_args:
        cmd.extend(default_args)
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, cwd=ROOT, env=_env(), check=True)


def run_script(relative_script: str, default_args: list[str] | None = None) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / relative_script)]
    if default_args:
        cmd.extend(default_args)
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, cwd=ROOT, env=_env(), check=True)
