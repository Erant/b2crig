"""b2crunner as b2crig uses it: separate processes and files, never an import.

- Its tools (b2crunner docs/tools.md: WAN, Sapiens2 seg and pointmaps, SeedVR2, SAM-3D-Body video fits, the MHR subject
  export, body recovery) run through its launcher, `tools/run.py`, which picks the environment each needs.
- Its main venv (body2colmap, pyrender) runs b2crig's own drawing scripts (tools/draw_control.py, the render half of
  tools/pose_reference.py).
- The delivered splats' header records are read by b2crig/plyheader.py from its format spec.

The checkout is $B2CRUNNER (default ~/Projects/b2crunner). A host without some of its environments points the
launcher elsewhere with B2CRUNNER_PYTHON_<ENV> / B2CRUNNER_PATH_<ENV> (docs/tools.md).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

B2CRUNNER = Path(os.environ.get("B2CRUNNER", Path.home() / "Projects" / "b2crunner"))
MAIN_PYTHON = B2CRUNNER / ".venv" / "bin" / "python"


def tool_cmd(name: str, *args) -> list[str]:
    """The command line that runs b2crunner's tools/<name>.py in its environment."""
    return [sys.executable, str(B2CRUNNER / "tools" / "run.py"), name, *(str(a) for a in args)]


def tool(name: str, *args, **kw) -> subprocess.CompletedProcess:
    """Run a b2crunner tool (check=True unless given); `kw` goes to subprocess.run (stdout, env, ...)."""
    kw.setdefault("check", True)
    return subprocess.run(tool_cmd(name, *args), **kw)
