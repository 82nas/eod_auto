#!/usr/bin/env python3
# handover.py – 인수인계 v1.1.5 (Serializer.save_state 구문 최종 수정 및 경로 처리 개선)

from __future__ import annotations
import os, sys, datetime, json, textwrap, pathlib, shutil, traceback, argparse, hashlib, importlib
from typing import List, Dict, Tuple, Optional, Type, Any
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 의존성 로드
# ─────────────────────────────────────────────────────────────────────────────
try:
    from git import Repo, GitCommandError, InvalidGitRepositoryError, Blob, Commit
    import requests
    from rich import print, box
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.console import Console
    from rich.syntax import Syntax
    import markdown2
except ImportError as e:
    print(f"[bold red]오류: 라이브러리 미설치[/]\n{e}")
    print("→ pip install gitpython requests rich python-dotenv markdown2")
    sys.exit(1)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 경로 & 상수
# ─────────────────────────────────────────────────────────────────────────────
APP_ROOT = pathlib.Path(".").resolve()
STATE_DIR = APP_ROOT / "ai_states"
ART_DIR = APP_ROOT / "artifacts"
BACKEND_DIR = pathlib.Path(__file__).resolve().parent / "backends"
COMMIT_TAG = "state("

# ─────────────────────────────────────────────────────────────────────────────
# AI 백엔드 로딩
# ─────────────────────────────────────────────────────────────────────────────
try:
    spec = importlib.util.spec_from_file_location(
        "backends.base", BACKEND_DIR / "base.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError
    _base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_base)
    AIBaseBackend = _base.AIBaseBackend
except Exception as e:
    print(f"[bold red]backends.base 임포트 실패: {e}[/]")
    sys.exit(1)

available_backends: Dict[str, Type[AIBaseBackend]] = {}
if BACKEND_DIR.exists():
    for f in BACKEND_DIR.glob("*.py"):
        if f.stem in {"__init__", "base"}:
            continue
        spec = importlib.util.spec_from_file_location(f"backends.{f.stem}", f)
        if spec and spec.loader:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            for v in m.__dict__.values():
                if (
                    isinstance(v, type)
                    and issubclass(v, AIBaseBackend)
                    and v is not AIBaseBackend
                ):
                    available_backends[v.get_name()] = v


# ─────────────────────────────────────────────────────────────────────────────
# AIProvider
# ─────────────────────────────────────────────────────────────────────────────
class AIProvider:
    def __init__(self, name: str, cfg: Dict[str, Any]):
        if name == "none":
            self.backend = None
            return
        if name not in available_backends:
            raise ValueError(f"알 수 없는 백엔드: {name}")
        self.backend = available_backends[name](cfg)

    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        if not self.backend:
            raise RuntimeError("AI 백엔드 off")
        return self.backend.make_summary(task, ctx, arts)

    def verify_summary(self, md: str) -> Tuple[bool, str]:
        if not self.backend:
            raise RuntimeError("AI 백엔드 off")
        return self.backend.verify_summary(md)

    def load_report(self, md: str) -> str:
        if not self.backend:
            raise RuntimeError("AI 백엔드 off")
        return self.backend.load_report(md)


# ─────────────────────────────────────────────────────────────────────────────
# GitRepo
# ─────────────────────────────────────────────────────────────────────────────
class GitRepo:
    def __init__(self, path: pathlib.Path):
        self.repo = Repo(path)

    def _run(self, cmd, *a):
        try:
            return cmd(*a)
        except GitCommandError as e:
            raise RuntimeError(e.stderr.strip()) from e

    def branch(self) -> str:
        try:
            return self.repo.active_branch.name
        except Exception:
            return f"DETACHED@{self.repo.head.commit.hexsha[:7]}"

    def save(
        self, paths: List[pathlib.Path], task: str, snap: Optional[pathlib.Path]
    ) -> str:
        to_add = [str(p) for p in paths] + ([str(snap)] if snap else [])
        self._run(self.repo.git.add, *to_add)
        self._run(self.repo.index.commit, f"{COMMIT_TAG}{task})")
        if self.repo.remotes:
            br = self.branch()
            if not br.startswith("DETACHED"):
                try:
                    self._run(self.repo.git.push, "origin", br)
                    print("[green]원격 푸시 완료[/]")
                except RuntimeError as e:
                    print(f"[yellow]원격 푸시 실패: {e}[/]")
        return self.repo.head.commit.hexsha[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Serializer (indent bug fixed)
# ─────────────────────────────────────────────────────────────────────────────
class Serializer:
    @staticmethod
    def _sha(fp: pathlib.Path) -> Optional[str]:
        h = hashlib.sha256()
        try:
            with fp.open("rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()
        except IOError:
            return None

    @staticmethod
    def save_state(
        md: str, task: str, state_dir: pathlib.Path, art_dir: pathlib.Path
    ) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        safe = (
            "".join(c for c in task if c.isalnum() or c in " _-")
            .strip()
            .replace(" ", "_")
            or "task"
        )
        base = f"{ts}_{safe}"

        state_dir.mkdir(exist_ok=True)
        art_dir.mkdir(exist_ok=True)

        md_f = state_dir / f"{base}.md"
        meta_f = state_dir / f"{base}.meta.json"
        md_f.write_text(md, encoding="utf-8")

        snap_dir, checksums = None, {}
        arts = [f for f in art_dir.iterdir() if f.is_file()]
        if arts:
            snap_dir = art_dir / f"{base}_artifacts"
            snap_dir.mkdir(parents=True, exist_ok=True)
            for f_art in arts:
                try:
                    dest = snap_dir / f_art.name
                    shutil.copy2(f_art, dest)
                    cs = Serializer._sha(dest)
                    if cs:
                        checksums[f_art.name] = cs
                except Exception as e:
                    print(f"[yellow]아티팩트 복사 실패 {f_art.name}: {e}[/]")

        meta = {"task": task, "ts": ts, "artifact_checksums": checksums}
        meta_f.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return [md_f, meta_f], (
            snap_dir if snap_dir and any(snap_dir.iterdir()) else None
        )


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
class UI:
    console = Console()
    ask = staticmethod(lambda t, d="": Prompt.ask(f"[cyan]{t}[/]", default=d))


# ─────────────────────────────────────────────────────────────────────────────
# Handover
# ─────────────────────────────────────────────────────────────────────────────
class Handover:
    def __init__(self, backend: str):
        self.git = GitRepo(APP_ROOT)
        self.ai = AIProvider(backend, {})

    def save(self):
        task = UI.ask("작업 이름", self.git.branch())
        ctx = UI.ask("작업 내용 요약")
        md = self.ai.make_summary(task, ctx, [])
        paths, snap = Serializer.save_state(md, task, STATE_DIR, ART_DIR)
        commit = self.git.save(paths, task, snap)
        UI.console.print(f"[bold green]저장 OK: {commit}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend", choices=list(available_backends) + ["none"], default="none"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("save")
    args = p.parse_args()

    if not (APP_ROOT / ".git").exists():
        print("❌ Git 저장소에서 실행하세요")
        sys.exit(1)

    if args.cmd == "save":
        Handover(args.backend).save()


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("Python 3.8+ 필요")
        sys.exit(1)
    main()
