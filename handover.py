#!/usr/bin/env python3
# handover.py – 인수인계 v1.1.4-fix2  (문법 오류만 패치, 로직 그대로)

from __future__ import annotations
import os, sys, datetime, json, pathlib, shutil, traceback, argparse, hashlib, importlib
from typing import List, Dict, Tuple, Optional, Type, Any
from dotenv import load_dotenv

# ────────── 의존성 로드 ───────────────────────────────────────────────────────
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
    print(f"[bold red]오류: 필요한 라이브러리가 설치되지 않았습니다.[/]\n{e}")
    print("→ pip install gitpython requests rich python-dotenv markdown2")
    sys.exit(1)

load_dotenv()

# ────────── 경로 & 상수 ───────────────────────────────────────────────────────
INITIAL_ROOT_PATH = pathlib.Path(".").resolve()
STATE_DIR_MODULE_LEVEL = INITIAL_ROOT_PATH / "ai_states"
ART_DIR_MODULE_LEVEL = INITIAL_ROOT_PATH / "artifacts"
BACKEND_DIR = pathlib.Path(__file__).resolve().parent / "backends"
COMMIT_TAG = "state("

# ────────── AI 백엔드 로딩 ────────────────────────────────────────────────────
try:
    spec = importlib.util.spec_from_file_location("backends.base", BACKEND_DIR / "base.py")
    if spec is None or spec.loader is None:
        raise ImportError("backends.base 모듈 스펙을 찾을 수 없습니다.")
    _base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_base)
    AIBaseBackend = _base.AIBaseBackend
except Exception as e:
    print(f"[bold red]오류: backends.base 임포트 실패: {e}[/]")
    sys.exit(1)

available_backends: Dict[str, Type[AIBaseBackend]] = {}
if BACKEND_DIR.exists():
    for f_py in BACKEND_DIR.glob("*.py"):
        if f_py.stem in {"__init__", "base"}:
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"backends.{f_py.stem}", f_py)
            if spec and spec.loader:
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                for obj in m.__dict__.values():
                    if isinstance(obj, type) and issubclass(obj, AIBaseBackend) and obj is not AIBaseBackend:
                        n = obj.get_name()
                        if n != "base":
                            available_backends[n] = obj
        except Exception as e:
            print(f"[yellow]경고: {f_py.name} 로딩 중 예외: {e}[/]")

# ────────── AI Provider ──────────────────────────────────────────────────────
class AIProvider:
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        if backend_name == "none":
            self.backend = None
            print("[dim]AI 백엔드: none (비활성)[/]")
            return
        if backend_name not in available_backends:
            raise ValueError(f"알 수 없는 백엔드: {backend_name}")
        self.backend: AIBaseBackend = available_backends[backend_name](config)
        print(f"[dim]AI 백엔드 사용: {backend_name}[/]")

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

# ────────── GitRepo ─────────────────────────────────────────────────────────
class GitRepo:
    def __init__(self, path: pathlib.Path):
        self.repo = Repo(path)

    def _run(self, func, *a):
        try:
            return func(*a)
        except GitCommandError as e:
            raise RuntimeError(e.stderr.strip()) from e

    def get_current_branch(self) -> str:
        try:
            return self.repo.active_branch.name
        except Exception:
            return f"DETACHED_HEAD@{self.repo.head.commit.hexsha[:7]}"

    def save(self, paths: List[pathlib.Path], task: str, snap: Optional[pathlib.Path]) -> str:
        add_list = [str(p) for p in paths]
        if snap:
            add_list.append(str(snap))
        self._run(self.repo.git.add, *add_list)
        msg = f"{COMMIT_TAG}{task})"
        try:
            self._run(self.repo.index.commit, msg)
        except RuntimeError as e:
            if "nothing to commit" in str(e).lower():
                return self.repo.head.commit.hexsha[:8] + " (변경 없음)"
            raise
        if self.repo.remotes:
            br = self.get_current_branch()
            if not br.startswith("DETACHED_HEAD"):
                try:
                    self._run(self.repo.git.push, "origin", br)
                    print("[green]원격 푸시 완료[/]")
                except RuntimeError as e:
                    print(f"[yellow]원격 푸시 실패: {e}[/]")
        return self.repo.head.commit.hexsha[:8]

# ────────── Serializer ──────────────────────────────────────────────────────
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
    def save_state(md: str, task: str,
                   state_dir: pathlib.Path,
                   art_dir: pathlib.Path,
                   root: pathlib.Path) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        safe = "".join(c for c in task if c.isalnum() or c in " _-").strip().replace(" ", "_") or "task"
        base = f"{ts}_{safe}"
        state_dir.mkdir(exist_ok=True)
        art_dir.mkdir(exist_ok=True)

        md_f = state_dir / f"{base}.md"
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

        meta_f = state_dir / f"{base}.meta.json"
        meta = {"task": task, "ts": ts, "artifact_checksums": checksums}
        meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        paths = [md_f, meta_f]
        return paths, snap_dir if snap_dir and any(snap_dir.iterdir()) else None

# ────────── UI ──────────────────────────────────────────────────────────────
class UI:
    console = Console()

    @staticmethod
    def ask(txt, default=""):  # 간단화
        return Prompt.ask(f"[bold cyan]{txt}[/]", default=default)

# ────────── Handover (save만 구현) ───────────────────────────────────────────
class Handover:
    def __init__(self, root: pathlib.Path, backend: str):
        self.root = root
        self.state = root / "ai_states"
        self.art = root / "artifacts"
        self.git = GitRepo(root)
        self.ai = AIProvider(backend, {})

    def save(self):
        task = UI.ask("작업 이름", self.git.get_current_branch())
        ctx = UI.ask("작업 내용 요약")
        md = self.ai.make_summary(task, ctx, [])
        paths, snap = Serializer.save_state(md, task, self.state, self.art, self.root)
        commit = self.git.save(paths, task, snap)
        UI.console.print(f"[bold green]저장 및 커밋 완료: {commit}[/]")

# ────────── main_cli ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=list(available_backends) + ["none"], default="none")
    parser.add_argument("cmd", choices=["save"])
    args = parser.parse_args()

    root = pathlib.Path(".").resolve()
    if not (root / ".git").exists():
        print("Git 저장소에서 실행하세요."); sys.exit(1)

    h = Handover(root, args.backend)
    if args.cmd == "save":
        h.save()

if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("Python 3.8+ 필요"); sys.exit(1)
    main()
