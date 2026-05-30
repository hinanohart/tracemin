#!/usr/bin/env python3
"""Per-phase structural gate for the tracemin autonomous build (S0-S11).

This is a *lightweight structural* verifier: it checks that the artifacts a phase
is supposed to produce exist and are internally consistent. Heavy behavioural
checks (pytest / mypy / ruff) run separately in the build loop; this script is the
cheap, deterministic gate that the protocol calls at each phase boundary.

Usage:
    python scripts/verify_step.py S0_5          # run S0.5 checks, exit 0/1
    python scripts/verify_step.py S2 --dry-run  # list S2 checks without running
Exit codes: 0 = all checks pass, 1 = a check failed, 2 = no checks defined.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tracemin"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _module_has(module_rel: str, *symbols: str) -> bool:
    """True if src/tracemin/<module_rel> exists and its source defines every symbol."""
    text = _read(SRC / module_rel)
    if not text:
        return False
    return all((f"def {s}" in text) or (f"class {s}" in text) for s in symbols)


# --- S0.5: scaffold -----------------------------------------------------------


def check_license() -> bool:
    return "MIT License" in _read(ROOT / "LICENSE")


def check_notice() -> bool:
    n = _read(ROOT / "NOTICE")
    return "passwedge" in n and "context-sieve" in n and "seedloop" in n


def check_pyproject() -> bool:
    p = ROOT / "pyproject.toml"
    if not p.exists():
        return False
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    proj = data.get("project", {})
    deps = proj.get("dependencies", [])
    extras = proj.get("optional-dependencies", {})
    core_ok = proj.get("name") == "tracemin" and any("huggingface_hub" in d for d in deps)
    # core must NOT pull numpy/scipy/passwedge
    core_clean = not any(any(b in d for b in ("numpy", "scipy", "passwedge")) for d in deps)
    extras_ok = {"stochastic", "sieve", "seedloop", "openhands", "all"} <= set(extras)
    script_ok = data.get("project", {}).get("scripts", {}).get("tracemin") == "tracemin.cli:main"
    return bool(core_ok and core_clean and extras_ok and script_ok)


def check_import() -> bool:
    init = SRC / "__init__.py"
    if not init.exists():
        return False
    spec = importlib.util.spec_from_file_location("tracemin", init)
    if spec is None or spec.loader is None:
        return False
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return bool(getattr(mod, "__version__", "").startswith("0.1.0a"))


def check_layout() -> bool:
    return (SRC / "adapters").is_dir() and (SRC / "bench").is_dir() and (ROOT / "tests").is_dir()


# --- S1: types / contract -----------------------------------------------------


def check_atoms() -> bool:
    return _module_has("atoms.py", "Atom") and "produces" in _read(SRC / "atoms.py")


def check_replay_contract() -> bool:
    t = _read(SRC / "replay.py")
    return _module_has("replay.py", "ReplayResult") and "Verdict" in t


def check_oracle() -> bool:
    return _module_has("oracle.py", "Oracle") and "classify" in _read(SRC / "oracle.py")


def check_signature() -> bool:
    return _module_has("signature.py", "failure_signature")


# --- S2: engine ---------------------------------------------------------------


def check_engine() -> bool:
    t = _read(SRC / "engine.py")
    return "def ddmin" in t or "def minimize" in t


def check_cache() -> bool:
    return (SRC / "cache.py").exists()


# --- S3: artifact + scrub -----------------------------------------------------


def check_artifact() -> bool:
    t = _read(SRC / "artifact.py")
    return "tracemin-repro/1" in t and "wf-constrained" in t


def check_scrub() -> bool:
    return _module_has("scrub.py", "scrub")


# --- S4: adapters -------------------------------------------------------------


def check_adapters() -> bool:
    a = SRC / "adapters"
    return all((a / f).exists() for f in ("hf.py", "openhands.py", "claude.py", "_render.py"))


# --- S5: cli + stochastic -----------------------------------------------------


def check_cli() -> bool:
    return _module_has("cli.py", "main")


def check_stochastic() -> bool:
    return (SRC / "stochastic.py").exists()


# --- S6: bench ----------------------------------------------------------------


def check_bench() -> bool:
    return (SRC / "bench" / "inject.py").exists() and (SRC / "bench" / "metrics.py").exists()


# --- S7: doc honesty gates ----------------------------------------------------


def check_doc_gate_scripts() -> bool:
    a = ROOT / "scripts" / "check_banned_phrases.sh"
    b = ROOT / "scripts" / "check_numeric_markers.sh"
    return a.exists() and os.access(a, os.X_OK) and b.exists() and os.access(b, os.X_OK)


def check_readme_filled() -> bool:
    t = _read(ROOT / "README.md")
    return "@S7" not in t and "@S6" not in t and "MARKER@" not in t


STEP_CHECKS: dict[str, list[tuple[str, object]]] = {
    "S0_5": [
        ("LICENSE is MIT", check_license),
        ("NOTICE credits passwedge/context-sieve/seedloop", check_notice),
        ("pyproject: name+core dep huggingface_hub only, 5 extras, script entry", check_pyproject),
        ("import tracemin works, __version__ 0.1.0a*", check_import),
        ("src layout (adapters/, bench/, tests/)", check_layout),
    ],
    "S1": [
        ("atoms.py defines Atom with produces/requires", check_atoms),
        ("replay.py defines ReplayResult + Verdict", check_replay_contract),
        ("oracle.py defines Oracle.classify", check_oracle),
        ("signature.py defines failure_signature", check_signature),
    ],
    "S2": [
        ("engine.py defines ddmin/minimize", check_engine),
        ("cache.py exists", check_cache),
    ],
    "S3": [
        ("artifact.py schema tracemin-repro/1 + wf-constrained", check_artifact),
        ("scrub.py defines scrub()", check_scrub),
    ],
    "S4": [
        ("adapters hf/openhands/claude/_render present", check_adapters),
    ],
    "S5": [
        ("cli.py defines main()", check_cli),
        ("stochastic.py exists", check_stochastic),
    ],
    "S6": [
        ("bench inject.py + metrics.py present", check_bench),
    ],
    "S7": [
        ("doc-gate scripts exist + executable", check_doc_gate_scripts),
        ("README has no @S6/@S7 placeholders left", check_readme_filled),
    ],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    step = args.step.replace(".", "_")
    checks = STEP_CHECKS.get(step)
    if checks is None:
        print(f"[verify_step] no checks defined for {args.step}", file=sys.stderr)
        return 2
    if args.dry_run:
        for desc, _ in checks:
            print(f"  [ ] {desc}")
        return 0
    ok = True
    for desc, fn in checks:
        try:
            passed = bool(fn())  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - report any check crash as failure
            passed = False
            desc = f"{desc}  (raised {type(exc).__name__}: {exc})"
        print(f"  [{'x' if passed else ' '}] {desc}")
        ok = ok and passed
    print(f"[verify_step] {args.step}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
