#!/usr/bin/env python3
# handover.py – 인수인계 v1.1 (무료 · 로컬 LLM 전용 + 확장 기능)
#
# 기능 추가 (v1.1):
# - AI 백엔드 플러그인 시스템 (--backend 옵션)
# - Git Diff 프리뷰 (diff 커맨드)
# - 아티팩트 스냅샷 체크섬 저장 (SHA-256)
# - HTML 리치 프리뷰 생성
# - 'save' 시 자동 입력 제안 (브랜치명, 커밋 로그)
# - 'load' 시 최신 상태 바로 로드 (--latest 옵션)
#
# 의존성:
# pip install gitpython requests rich python-dotenv markdown2
# (requirements.txt 참고)
#
# 설정:
# - .env 파일 또는 환경 변수: OLLAMA_BASE_URL, AI_MODEL, HF_API_KEY
# - CLI 옵션: --backend [ollama|huggingface|...]
#
# 개발 환경 권장:
# - pre-commit, ruff, black 사용 권장 (스크립트 코드 품질 관리용)
#   예: pip install pre-commit ruff black
#       pre-commit install
#       (Requires .pre-commit-config.yaml configuration)

from __future__ import annotations
import os
import sys
import datetime
import json
import textwrap
import pathlib
import shutil
import difflib
import traceback
import tempfile
import argparse
import hashlib
import importlib
from typing import List, Dict, Tuple, Optional, Type
from dotenv import load_dotenv

# --- 의존성 로드 ---
try:
    from git import Repo, GitCommandError, InvalidGitRepositoryError, Blob, Commit
    import requests
    from rich import print, box
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.console import Console
    from rich.syntax import Syntax
    import markdown2 # HTML 생성을 위해 markdown2 임포트
    # backends 모듈은 아래에서 동적으로 로드
except ImportError as e:
    print(f"[bold red]오류: 필요한 라이브러리가 설치되지 않았습니다.[/]\n{e}")
    print("팁: [yellow]pip install gitpython requests rich python-dotenv markdown2[/] 명령을 실행하세요.")
    sys.exit(1)

# --- 환경 변수 로드 ---
load_dotenv()

# --- 경로 & 상수 ---
try:
    ROOT = pathlib.Path(Repo('.', search_parent_directories=True).working_tree_dir)
except InvalidGitRepositoryError:
    print("[bold red]오류: 이 스크립트는 Git 저장소 내에서 실행되어야 합니다.[/]")
    # Git 저장소가 아니어도 초기 설정 스크립트는 실행될 수 있으므로,
    # 여기서는 일단 경로만 설정하고 실제 GitRepo 객체 생성 시 오류를 처리하도록 함
    # 혹은 argparse 전에 체크하도록 변경 고려
    print("[yellow]경고: 현재 폴더가 Git 저장소가 아닙니다. 'git init'을 실행했는지 확인하세요.[/]")
    # 일단 스크립트 실행은 계속하도록 하되, Git 기능 사용 시 문제 발생 가능
    ROOT = pathlib.Path('.').resolve() # 임시로 현재 경로 사용

STATE_DIR = ROOT / "ai_states"
ART_DIR = ROOT / "artifacts"
BACKEND_DIR = pathlib.Path(__file__).parent / "backends"
COMMIT_TAG = "state("

STATE_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)
# backends 디렉토리는 스크립트 로딩 후 체크

# --- AI 백엔드 로딩 ---
# Base class import path needs adjustment if handover.py is not at root relative to backends
try:
    # Assuming backends directory is sibling to handover.py
    if not BACKEND_DIR.is_dir():
         raise FileNotFoundError(f"Backend directory not found at {BACKEND_DIR}")

    # Add backend directory to sys.path if necessary (might not be needed if run as module)
    # sys.path.insert(0, str(pathlib.Path(__file__).parent))
    # print(f"Importing from: {BACKEND_DIR}")
    import backends.base # Import base class

    available_backends: Dict[str, Type[backends.base.AIBaseBackend]] = {}
    for f in BACKEND_DIR.glob("*.py"):
        module_name = f.stem
        if module_name == "__init__" or module_name == "base":
            continue
        try:
            # Dynamically import the module relative to the 'backends' package
            module = importlib.import_module(f"backends.{module_name}")
            for name, obj in module.__dict__.items():
                if (isinstance(obj, type) and
                        issubclass(obj, backends.base.AIBaseBackend) and
                        obj is not backends.base.AIBaseBackend):
                    backend_name = obj.get_name()
                    if backend_name != "base":
                        available_backends[backend_name] = obj
        except ImportError as e:
            print(f"[yellow]경고: 백엔드 모듈 로딩 실패 {f}: {e}[/]")
        except AttributeError as e:
             print(f"[yellow]경고: 백엔드 클래스 속성 오류 {f}: {e}[/]")


except FileNotFoundError as e:
     print(f"[bold red]오류: {e}. 'backends' 폴더와 그 안의 백엔드 파일(base.py 등)이 필요합니다.[/]")
     sys.exit(1)
except ImportError as e:
     print(f"[bold red]오류: 백엔드 모듈(backends.base 등)을 임포트할 수 없습니다. 경로 또는 __init__.py 파일을 확인하세요: {e}[/]")
     sys.exit(1)


if not available_backends:
    print("[bold red]오류: 사용 가능한 AI 백엔드를 찾을 수 없습니다. 'backends' 폴더와 그 안의 파일을 확인하세요.[/]")
    sys.exit(1)

# --- AI Provider (Uses selected backend) ---
class AIProvider:
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        if backend_name not in available_backends:
            raise ValueError(f"알 수 없는 백엔드: {backend_name}. 사용 가능: {list(available_backends.keys())}")

        BackendClass = available_backends[backend_name]
        try:
            self.backend: backends.base.AIBaseBackend = BackendClass(config)
            print(f"[dim]AI 백엔드 사용: [bold cyan]{backend_name}[/][/dim]")
        except Exception as e:
            print(f"[bold red]오류: 백엔드 '{backend_name}' 초기화 실패: {e}[/]")
            print(f"[yellow]필요 설정:\n{BackendClass.get_config_description()}[/]")
            raise e

    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        return self.backend.make_summary(task, ctx, arts)

    def verify_summary(self, md: str) -> Tuple[bool, str]:
        is_ok, msg = self.backend.verify_summary(md)
        # Basic structural check here as a fallback/complement
        if is_ok:
           lines = md.strip().split('\n')
           headers = [l.strip() for l in lines if l.startswith('#')]
           required_headers_prefix = ["#", "## 목표", "## 진행", "## 결정", "## 결과", "## 다음할일", "## 산출물"]
           if len(headers) < len(required_headers_prefix):
               is_ok = False
               msg = f"헤더 개수 부족 (필수 {len(required_headers_prefix)}개 시작 항목 필요, 발견 {len(headers)}개)"
           else:
               for i, req_prefix in enumerate(required_headers_prefix):
                  if i==0 and not headers[i].startswith("#"):
                      is_ok = False
                      msg = f"첫 헤더가 '#'로 시작하지 않음: '{headers[i]}'"
                      break
                  elif i > 0 and not headers[i].startswith(req_prefix):
                      is_ok = False
                      msg = f"헤더 순서 또는 레벨 오류: '{headers[i]}' (예상: '{req_prefix}' 시작)"
                      break
        return is_ok, msg

    def load_report(self, md: str) -> str:
        return self.backend.load_report(md)


# ─────────────── Git 래퍼 ────────────────
class GitRepo:
    def __init__(self, repo_path: pathlib.Path):
        try:
            self.repo = Repo(repo_path)
            if self.repo.bare:
                 raise InvalidGitRepositoryError(f"'{repo_path}'는 bare 저장소입니다.")
        except InvalidGitRepositoryError as e:
            # Allow script to continue if not a git repo for non-git commands?
            # Or raise error here? Let's raise for now.
            print(f"[bold red]오류: '{repo_path}'는 유효한 Git 작업 디렉토리가 아닙니다. 'git init'을 실행하세요.[/]")
            raise e

    def _safe(self, git_func, *args, **kwargs):
        try:
            return git_func(*args, **kwargs)
        except GitCommandError as e:
            stderr = e.stderr.strip()
            raise RuntimeError(f"Git 명령어 실패: {e.command}\n오류: {stderr}") from e

    def get_last_state_commit(self) -> Optional[Commit]:
         try:
             for commit in self.repo.iter_commits(max_count=200, first_parent=True): # Search more, avoid complex history
                 if commit.message.startswith(COMMIT_TAG):
                     return commit
         except Exception as e:
             print(f"[yellow]경고: 마지막 상태 커밋 조회 실패: {e}[/]")
         return None

    def get_commit_messages_since(self, commit_hash: Optional[str]) -> str:
        if not commit_hash:
            try:
                # Get recent commits not tagged as 'state' maybe? Or just recent N.
                commits = list(self.repo.iter_commits(max_count=10))
                # Filter out state commits themselves? No, show all recent work.
                log = "\n".join(f"- {c.hexsha[:7]}: {c.summary}" for c in reversed(commits))
                return f"최근 커밋 {len(commits)}개:\n{log}" if log else "최근 커밋 없음"
            except Exception as e:
                 print(f"[yellow]경고: 최근 커밋 로그 조회 실패: {e}[/]")
                 return "최근 커밋 로그를 가져올 수 없습니다."

        try:
            log_cmd = f"{commit_hash}..HEAD"
            commit_log = self.repo.git.log(log_cmd, '--pretty=format:- %h: %s', '--abbrev-commit', '--no-merges')
            return f"'{commit_hash[:8]}' 이후 커밋:\n{commit_log}" if commit_log else f"'{commit_hash[:8]}' 이후 커밋 없음"
        except GitCommandError as e:
            # Handle case where commit_hash might not be found or other git log errors
            print(f"[yellow]경고: 커밋 로그 조회 실패 ({log_cmd}): {e.stderr}[/]")
            return f"'{commit_hash[:8]}' 이후 커밋 로그를 가져올 수 없습니다."
        except Exception as e:
            print(f"[yellow]경고: 커밋 로그 조회 중 오류: {e}[/]")
            return f"'{commit_hash[:8]}' 이후 커밋 로그를 가져올 수 없습니다."

    def get_current_branch(self) -> Optional[str]:
        try:
            # repo.active_branch might fail in detached HEAD state
            return self.repo.active_branch.name
        except TypeError: # Handle detached HEAD
            # Get the current commit hash in detached state
            try:
                return f"DETACHED_HEAD@{self.repo.head.commit.hexsha[:7]}"
            except Exception:
                 return "DETACHED_HEAD"
        except Exception as e:
            print(f"[yellow]경고: 현재 브랜치 이름 조회 실패: {e}[/]")
            return None

    def get_diff(self, target: str = "HEAD", color: bool = True) -> str:
        try:
            # Ensure target is valid before diffing
            try:
                self.repo.commit(target) # Check if target commit exists
            except Exception:
                 # Check if target is a branch/ref
                 if target not in self.repo.refs:
                     raise ValueError(f"비교 대상 '{target}'을 찾을 수 없습니다 (커밋 또는 브랜치).")

            color_opt = '--color=always' if color else '--color=never'
            # Diff index (staged) vs target
            staged_diff = self.repo.git.diff('--staged', target, color=color_opt)
            # Diff working tree vs index (unstaged changes relative to what's staged)
            unstaged_diff = self.repo.git.diff(color=color_opt)

            diff_output = ""
            if staged_diff:
                diff_output += f"--- Staged Changes (vs {target}) ---\n{staged_diff}\n\n"
            if unstaged_diff:
                 # Note: This diffs working tree vs index, not vs target directly
                 # To diff working tree vs target: repo.git.diff(target, color=color_opt)
                 # Let's show diff working tree vs target for clarity
                 working_tree_vs_target_diff = self.repo.git.diff(target, color=color_opt)
                 if working_tree_vs_target_diff and working_tree_vs_target_diff != staged_diff: # Avoid showing same diff twice if only staged changes
                     diff_output += f"--- Unstaged Changes (vs {target}) ---\n{working_tree_vs_target_diff}\n"
                 elif not staged_diff and working_tree_vs_target_diff: # Only unstaged exist
                      diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"

            return diff_output if diff_output else f"'{target}'과(와) 변경 사항 없음 (Git 추적 파일 기준)"

        except GitCommandError as e:
            return f"Diff 생성 오류: {e.stderr}"
        except ValueError as e: # Handle invalid target
             return f"Diff 생성 오류: {e}"
        except Exception as e:
             return f"Diff 생성 중 알 수 없는 오류: {e}"

    def save(self, state_paths: List[pathlib.Path], task: str, snapshot_dir: Optional[pathlib.Path]) -> str:
        paths_to_add = [str(p.resolve()) for p in state_paths]
        if snapshot_dir and snapshot_dir.exists():
            # Check if snapshot dir is empty before adding
            if any(snapshot_dir.iterdir()):
                paths_to_add.append(str(snapshot_dir.resolve()))
            else:
                 print("[dim]빈 스냅샷 디렉토리는 Git에 추가하지 않습니다.[/]")
                 snapshot_dir = None # Treat as if no snapshot was added

        if not state_paths: # Must have at least state files
             raise ValueError("저장할 상태 파일이 없습니다.")

        # Check for changes before adding/committing
        # Note: This check might be complex if only metadata changed slightly
        # For simplicity, just add and commit. Git handles no-change commits.

        self._safe(self.repo.git.add, *paths_to_add)

        # Check if staging area is actually different from HEAD before committing
        if not self.repo.index.diff("HEAD"):
             # Check if only untracked files were added (like snapshot dir)
             # This logic is tricky; let's allow commit even if only untracked added
             print("[yellow]경고: 이전 커밋과 비교하여 변경된 내용이 없습니다 (스테이징 기준).[/]")
             # Still proceed to commit, maybe only untracked files were added

        commit_message = f"{COMMIT_TAG}{task})"
        self._safe(self.repo.index.commit, commit_message)

        # Push only if remote is configured
        remote_push_ok = False
        if self.repo.remotes:
            try:
                # Push current branch to its upstream or default remote (origin)
                # This might need more specific branch handling
                self._safe(self.repo.git.push)
                print("[green]원격 저장소에 푸시 완료.[/]")
                remote_push_ok = True
            except RuntimeError as e:
                print(f"[yellow]경고: 원격 저장소 푸시 실패. 로컬에는 커밋되었습니다. ({e})[/]")
        else:
            print("[yellow]경고: 설정된 원격 저장소가 없어 푸시를 건너<0xEB><0x9B><0x8D>니다.[/]")


        commit_hash = self.repo.head.commit.hexsha[:8]
        return commit_hash

    def list_states(self) -> List[Dict]:
        items = []
        try:
            # Look further back if needed, handle potential performance issue
            commits = list(self.repo.iter_commits(max_count=100, first_parent=True))
        except Exception as e:
            raise RuntimeError(f"Git 커밋 목록 조회 중 오류: {e}") from e

        for commit in commits:
            if not commit.message.startswith(COMMIT_TAG): continue

            meta_blob: Optional[Blob] = None
            headline = ""
            meta_path_prefix = STATE_DIR.name # Relative path to state dir

            try:
                # Search within the specific commit's tree
                possible_meta_paths = [item.path for item in commit.tree.traverse() if isinstance(item, Blob) and item.path.startswith(meta_path_prefix) and item.path.endswith(".meta.json")]

                if possible_meta_paths:
                    # Assume the first one found is the relevant one for this commit
                    meta_path = possible_meta_paths[0]
                    meta_blob = commit.tree[meta_path]

                    try:
                        metadata = json.loads(meta_blob.data_stream.read().decode('utf-8'))
                        headline = metadata.get("headline", "")
                    except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e_parse:
                        print(f"[yellow]경고: 커밋 {commit.hexsha[:8]} 메타데이터 파싱 실패 ({meta_path}, {e_parse})[/]")
                        headline = "[메타데이터 오류]"
                    except Exception as e_read:
                         print(f"[yellow]경고: 커밋 {commit.hexsha[:8]} 메타데이터 읽기 실패 ({meta_path}, {e_read})[/]")
                         headline = "[메타데이터 읽기 오류]"
                # else: No meta file found in this commit's state dir

            except Exception as e_traverse:
                 print(f"[yellow]경고: 커밋 {commit.hexsha[:8]} 트리 탐색 중 오류 ({e_traverse})[/]")


            items.append({
                "hash": commit.hexsha[:8],
                "task": commit.message[len(COMMIT_TAG):-1].strip(),
                "time": datetime.datetime.fromtimestamp(commit.committed_date).strftime("%Y-%m-%d %H:%M"),
                "head": headline or "-"
            })

        return list(reversed(items)) # Show oldest first or newest first? Newest first is reversed.

    def load_state(self, commit_hash: str) -> str:
        try:
            # Use rev_parse for more robust commit finding
            commit = self.repo.commit(self.repo.git.rev_parse(commit_hash))
        except GitCommandError:
             raise RuntimeError(f"커밋 '{commit_hash}'를 찾을 수 없습니다.") from None
        except Exception as e:
             raise RuntimeError(f"커밋 '{commit_hash}' 접근 중 오류: {e}") from e

        md_path_prefix = STATE_DIR.name
        possible_md_paths = [item.path for item in commit.tree.traverse() if isinstance(item, Blob) and item.path.startswith(md_path_prefix) and item.path.endswith(".md")]

        if not possible_md_paths:
             raise RuntimeError(f"커밋 '{commit_hash}'에서 상태 파일(.md)을 찾을 수 없습니다 ({md_path_prefix} 폴더 내부).")

        # Assume first md file found in state dir is the one
        md_path = possible_md_paths[0]
        try:
             md_blob = commit.tree[md_path]
             return md_blob.data_stream.read().decode('utf-8')
        except KeyError: # Should not happen if path came from traverse, but check anyway
             raise RuntimeError(f"커밋 '{commit_hash}'에서 상태 파일 경로 '{md_path}'를 찾을 수 없습니다.") from None
        except Exception as e:
             raise RuntimeError(f"커밋 '{commit_hash}'에서 상태 파일 '{md_path}' 로드 실패: {e}") from e


# ─────────────── 직렬화 + HTML/Checksum ────────────────
class Serializer:
    @staticmethod
    def _calculate_sha256(filepath: pathlib.Path) -> Optional[str]:
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while True:
                    byte_block = f.read(4096) # Read in blocks
                    if not byte_block:
                        break
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except IOError as e:
            print(f"[yellow]경고: 파일 해시 계산 실패 {filepath}: {e}[/]")
            return None # Indicate failure

    @staticmethod
    def _generate_html(md_content: str, title: str) -> str:
         # Basic embedded CSS
         css = """
<style>
 body { font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; color: #333; }
 h1, h2 { border-bottom: 1px solid #eee; padding-bottom: 0.3em; margin-top: 1.5em; margin-bottom: 1em; }
 h1 { font-size: 2em; }
 h2 { font-size: 1.5em; }
 ul, ol { padding-left: 2em; }
 li { margin-bottom: 0.5em; }
 code { background-color: #f0f0f0; padding: 0.2em 0.4em; border-radius: 3px; font-family: monospace; font-size: 0.9em; }
 pre { background-color: #f5f5f5; padding: 1em; border-radius: 4px; overflow-x: auto; }
 pre code { background-color: transparent; padding: 0; border-radius: 0; }
 blockquote { border-left: 4px solid #ccc; padding-left: 1em; color: #666; margin-left: 0; }
 table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
 th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
 th { background-color: #f2f2f2; }
</style>
"""
         # Use markdown2 with extras
         html_body = markdown2.markdown(
             md_content,
             extras=["metadata", "fenced-code-blocks", "tables", "strike", "task_list", "code-friendly"]
         )

         title_from_meta = title # Default
         if hasattr(html_body, "metadata") and html_body.metadata.get("title"):
             title_from_meta = html_body.metadata["title"]

         full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>{title_from_meta}</title>
 {css}
</head>
<body>
 {html_body}
</body>
</html>"""
         return full_html

    @staticmethod
    def save_state(md: str, task: str) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        # Generate safer filename (allow underscore, dash, space)
        safe_task_name = "".join(c for c in task if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        if not safe_task_name: safe_task_name = "untitled" # Fallback
        base_filename = f"{ts}_{safe_task_name}"

        state_file = STATE_DIR / f"{base_filename}.md"
        html_file = STATE_DIR / f"{base_filename}.html"
        meta_file = STATE_DIR / f"{base_filename}.meta.json"

        # 1. Save MD
        try:
            state_file.write_text(md, encoding="utf-8")
        except IOError as e:
            raise RuntimeError(f"상태 파일 저장 실패 ({state_file}): {e}") from e

        # 2. Generate and Save HTML
        html_saved = False
        try:
            html_content = Serializer._generate_html(md, task)
            html_file.write_text(html_content, encoding="utf-8")
            html_saved = True
        except Exception as e:
             print(f"[yellow]경고: HTML 프리뷰 생성/저장 실패 ({html_file}): {e}[/]")
             # Ensure partial file is removed if write failed midway (though write_text is often atomic)
             if html_file.exists(): html_file.unlink(missing_ok=True)

        # 3. Artifact Snapshot and Checksums
        snapshot_dir: Optional[pathlib.Path] = None
        artifact_checksums: Dict[str, str] = {}
        try:
             # Ensure ART_DIR exists before iterating
             ART_DIR.mkdir(exist_ok=True)
             artifact_files = [f for f in ART_DIR.iterdir() if f.is_file()]
        except OSError as e:
            print(f"[yellow]경고: 아티팩트 디렉토리 접근 불가 ({ART_DIR}): {e}. 스냅샷 건너<0xEB><0x9B><0x8D>.[/]")
            artifact_files = []


        if artifact_files:
            snapshot_dir = ART_DIR / f"{base_filename}_artifacts"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            print(f"[dim]아티팩트 스냅샷 생성 및 체크섬 계산 중... ({len(artifact_files)}개 파일)[/]")
            try:
                for f in artifact_files:
                    target_path = snapshot_dir / f.name
                    try:
                         shutil.copy2(f, target_path)
                         checksum = Serializer._calculate_sha256(target_path)
                         if checksum: # Only add if hashing was successful
                             artifact_checksums[f.name] = checksum
                    except (IOError, shutil.Error) as copy_err:
                         print(f"[yellow]경고: 아티팩트 파일 복사/해시 실패 ({f.name}): {copy_err}. 스냅샷에서 제외.[/]")
                         # Attempt to remove partially copied file? Risky. Let it be.
            except Exception as e_snap: # Catch broader errors during snapshot process
                 # Don't raise, just warn and continue without snapshot if possible
                 print(f"[red]오류: 아티팩트 스냅샷 생성 중 문제 발생: {e_snap}[/]")
                 # Attempt cleanup? Maybe just leave partial snapshot.
                 # snapshot_dir = None # Indicate snapshot failed? Or keep partial?
                 # artifact_checksums = {} # Clear checksums if snapshot failed

            if artifact_checksums: # Only print if checksums were generated
                 print(f"[dim]스냅샷 및 체크섬 완료: {snapshot_dir}[/]")
            elif snapshot_dir.exists() and not any(snapshot_dir.iterdir()):
                 print("[dim]스냅샷 디렉토리가 생성되었으나 복사/해시된 파일이 없습니다.[/]")
                 # Remove empty snapshot dir?
                 # try: snapshot_dir.rmdir() except OSError: pass
                 # snapshot_dir = None # Treat as no snapshot
            elif snapshot_dir.exists():
                 print("[yellow]경고: 스냅샷은 생성되었으나 체크섬 계산에 실패한 파일이 있습니다.[/]")


        else:
             print("[dim]복사할 아티팩트 파일이 없습니다.[/]")

        # 4. Save Metadata
        # Extract headline from first non-empty line starting with #
        headline = ""
        for line in md.splitlines():
            line_strip = line.strip()
            if line_strip.startswith("#"):
                 headline = line_strip.lstrip('# ').strip()
                 break
        if not headline: # Fallback if no header found
            headline = md.splitlines()[0].strip() if md.strip() else ""


        metadata = {
            "task": task,
            "ts": ts,
            "headline": headline,
            "artifact_checksums": artifact_checksums
        }
        try:
            meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except IOError as e:
            # Don't stop the whole process for meta failure, maybe? Or raise? Let's raise.
            raise RuntimeError(f"메타데이터 파일 저장 실패 ({meta_file}): {e}") from e

        paths_to_add = [state_file, meta_file]
        if html_saved: # Add HTML only if successfully created and saved
             paths_to_add.append(html_file)

        # Return snapshot dir only if it contains files that were checksummed?
        # Or just if it exists? Let's return if it exists and might contain files.
        valid_snapshot_dir = snapshot_dir if (snapshot_dir and snapshot_dir.exists() and any(snapshot_dir.iterdir())) else None

        return paths_to_add, valid_snapshot_dir

    @staticmethod
    def to_prompt(md: str, commit: str) -> str:
        # Make it clearer for AI processing
        return f"### CONTEXT START: State from Commit {commit} ###\n\n{md}\n\n### CONTEXT END ###"


# ─────────────── UI (Rich CLI) ───────────────
class UI:
    console = Console()

    @staticmethod
    def task_name(default: str = "작업 요약") -> str:
        return Prompt.ask("[bold cyan]현재 진행 중인 작업 이름[/]", default=default)

    @staticmethod
    def multiline(label: str, default: str = "") -> str:
        # Use Rich's Prompt for multiline input if available and suitable
        # Or stick to simpler input loop
        UI.console.print(f"\n[bold cyan]{label}[/]")
        UI.console.print("[dim](입력을 마치려면 빈 줄에서 Enter 키를 두 번 누르세요)[/]")
        lines = []
        if default:
             print("[dim]자동 제안된 내용 (편집 가능):[/]")
             print(Panel(default, border_style="dim", expand=False))
             print("[dim]--------------------[/]")
             # Don't pre-fill 'lines' - let user start fresh or copy-paste

        blank_count = 0
        while True:
            try:
                # Use console.input for better handling potentially
                line = input()
                if line == "":
                    blank_count += 1
                    if blank_count >= 2:
                        break
                else:
                    blank_count = 0
                    lines.append(line)
            except EOFError:
                break

        final_text = "\n".join(lines).strip()
        # If user entered nothing, and there was a default, maybe return default?
        # Current logic returns empty string if user enters nothing.
        # Let's return default if user input is empty AND default existed.
        if not final_text and default:
             print("[dim]입력이 없어 자동 제안된 내용으로 진행합니다.[/]")
             return default
        return final_text


    @staticmethod
    def notify(msg: str, style: str = "green"):
        UI.console.print(f"\n[bold {style}]✔ {msg}[/bold {style}]")

    @staticmethod
    def error(msg: str, details: Optional[str] = None):
        UI.console.print(f"\n[bold red]❌ 오류 발생: {msg}[/]")
        if details:
            # Limit traceback details length?
            details_short = "\n".join(details.splitlines()[-15:]) # Show last 15 lines
            UI.console.print(Panel(details_short, title="상세 정보 (Traceback)", border_style="dim red", expand=False))

    @staticmethod
    def pick_state(states: List[Dict]) -> Optional[str]:
        if not states:
            print("[yellow]저장된 상태가 없습니다.[/]")
            return None
        table = Table(title="[bold]저장된 인수인계 상태 목록[/]", box=box.ROUNDED, show_lines=True, expand=False)
        table.add_column("#", style="dim", justify="right")
        table.add_column("커밋 해시", style="cyan", no_wrap=True)
        table.add_column("작업 이름", style="magenta")
        table.add_column("저장 시각", style="green")
        table.add_column("요약 (Headline)", style="yellow")
        for i, s in enumerate(states): table.add_row(str(i), s["hash"], s["task"], s["time"], s["head"])
        UI.console.print(table)

        # Use Prompt with choices for better UX
        choices = [str(i) for i in range(len(states))]
        selection = Prompt.ask(
             "[bold cyan]로드할 상태의 번호 입력 (취소하려면 Enter)[/]",
             choices=choices + [""], # Allow empty input for cancel
             show_choices=False, # Don't show choices again
             default=""
        )

        if selection.isdigit() and 0 <= int(selection) < len(states):
            selected_hash = states[int(selection)]["hash"]
            print(f"[info]선택된 커밋: {selected_hash}[/]")
            return selected_hash
        else:
            print("[info]상태 로드를 취소했습니다.[/]")
            return None

    @staticmethod
    def panel(txt: str, title: str, border_style: str = "blue"):
        UI.console.print(Panel(txt, title=f"[bold]{title}[/]", border_style=border_style, expand=False, padding=(1, 2)))

    @staticmethod
    def diff_panel(diff_text: str, target: str):
        if not diff_text: return
        syntax = Syntax(diff_text, "diff", theme="default", line_numbers=False, word_wrap=False) # word_wrap=False for diffs
        UI.console.print(Panel(syntax, title=f"[bold]Diff Preview (vs {target})[/]", border_style="yellow", expand=True))


# ─────────────── Core + New Commands ───────────────
class Handover:
    def __init__(self, backend_choice: str):
        self.ui = UI()
        try:
             self.git = GitRepo(ROOT) # Init GitRepo first
        except Exception as e:
             # If Git init fails, maybe allow running without Git features?
             # For now, treat as critical failure.
             UI.error(f"Git 저장소 초기화/접근 실패 ({ROOT})", traceback.format_exc())
             sys.exit(1)
        try:
            self.ai = AIProvider(backend_name=backend_choice, config={})
        except Exception as e:
             UI.error(f"AI 백엔드 ({backend_choice}) 초기화 실패", traceback.format_exc())
             # Allow running without AI? No, AI is core.
             sys.exit(1)

    def save(self):
        try:
            # 1. Get Task Name (Default: Branch or last commit summary if detached)
            default_task = self.git.get_current_branch()
            if not default_task or default_task.startswith("DETACHED_HEAD"):
                try: default_task = self.git.repo.head.commit.summary
                except Exception: default_task = "작업 요약"
            task = self.ui.task_name(default=default_task)

            # 2. Get Context (Default: Commits since last state)
            last_state_commit = self.git.get_last_state_commit()
            last_commit_hash = last_state_commit.hexsha if last_state_commit else None
            default_context = self.git.get_commit_messages_since(last_commit_hash)
            context = self.ui.multiline("최근 작업 내용 또는 대화 요약 입력", default=default_context)
            if not context: # Require context
                 self.ui.error("작업 내용 요약이 비어있습니다. 저장을 취소합니다.")
                 return

            # 3. Artifacts & AI Summary Generation
            artifacts = []
            try:
                 artifacts = [f.name for f in ART_DIR.iterdir() if f.is_file()]
                 self.ui.console.print(f"[dim]현재 아티팩트: {', '.join(artifacts) or '없음'}[/]")
            except Exception as e:
                 self.ui.error(f"아티팩트 목록 조회 실패 ({ART_DIR}): {e}")

            self.ui.console.print("\n[bold yellow]AI가 인수인계 문서를 생성 중입니다...[/]")
            markdown_summary = self.ai.make_summary(task, context, artifacts)
            self.ui.panel(markdown_summary, "AI 생성 요약본 (검증 전)")

            # 4. AI Verification
            self.ui.console.print("[bold yellow]생성된 요약본을 AI가 검증 중입니다...[/]")
            is_ok, validation_msg = self.ai.verify_summary(markdown_summary)
            if not is_ok:
                # Maybe allow user to save anyway with a warning? For now, block.
                raise RuntimeError(f"AI 검증 실패:\n{validation_msg}")
            self.ui.notify("AI 검증 통과!", style="green")

            # 5. Save files & Snapshot
            state_files, snapshot_dir = Serializer.save_state(markdown_summary, task)

            # 6. Git Commit & Push
            commit_hash = self.git.save(state_files, task, snapshot_dir)
            self.ui.notify(f"인수인계 상태 저장 완료! (Commit: {commit_hash})", style="bold green")

            html_path = next((f for f in state_files if f.name.endswith(".html")), None)
            if html_path and html_path.exists():
                 self.ui.console.print(f"[dim]HTML 프리뷰 생성됨: {html_path.relative_to(ROOT)}[/]")

        except Exception as e:
            self.ui.error(str(e), traceback.format_exc())

    def load(self, latest: bool = False):
        try:
            saved_states = self.git.list_states()
            if not saved_states:
                self.ui.error("저장된 상태가 없습니다.") # Use error style
                return

            selected_hash: Optional[str] = None
            if latest:
                selected_hash = saved_states[-1]["hash"]
                print(f"[info]최신 상태 로드 중: {selected_hash} ({saved_states[-1]['task']})[/]")
            else:
                selected_hash = self.ui.pick_state(saved_states)

            if not selected_hash: return # User cancelled

            self.ui.console.print(f"[bold yellow]{selected_hash} 커밋에서 상태 정보를 로드 중입니다...[/]")
            markdown_content = self.git.load_state(selected_hash)
            prompt_format = Serializer.to_prompt(markdown_content, selected_hash)
            self.ui.panel(prompt_format, f"로드된 상태 (Commit: {selected_hash})", border_style="cyan")

            self.ui.console.print("[bold yellow]AI가 로드된 상태를 분석하고 이해도를 보고합니다...[/]")
            report = self.ai.load_report(markdown_content)
            self.ui.panel(report, "AI 이해도 보고서", border_style="magenta")

        except Exception as e:
            self.ui.error(str(e), traceback.format_exc())

    def diff(self, target: str = "HEAD"):
        try:
             self.ui.console.print(f"[bold yellow]'{target}' 대비 변경 사항을 확인 중입니다... (Git 추적 파일 기준)[/]")
             diff_output = self.git.get_diff(target, color=True)
             # Only show panel if there is diff output
             if diff_output and not diff_output.startswith("Diff 생성 오류") and "변경 사항 없음" not in diff_output:
                 self.ui.diff_panel(diff_output, target)
             else:
                  print(f"[dim]{diff_output}[/]") # Print no changes or error message plainly
        except Exception as e:
            self.ui.error(str(e), traceback.format_exc())

    def verify_checksums(self, commit_hash: str):
         # Placeholder implementation - shows stored checksums
         self.ui.console.print(f"[yellow]기능 구현 예정:[/][dim] 커밋 {commit_hash}의 아티팩트 체크섬 실제 파일과 비교 검증[/]")
         self.ui.console.print("[dim](현재는 저장 시 기록된 체크섬 정보만 보여줍니다)[/]")
         try:
             # Use rev_parse for robustness
             commit = self.git.repo.commit(self.git.repo.git.rev_parse(commit_hash))
             meta_blob = None
             meta_path_prefix = STATE_DIR.name
             possible_meta_paths = [item.path for item in commit.tree.traverse() if isinstance(item, Blob) and item.path.startswith(meta_path_prefix) and item.path.endswith(".meta.json")]

             if not possible_meta_paths:
                  self.ui.error(f"커밋 {commit_hash}에서 메타데이터 파일(.meta.json)을 찾을 수 없음.")
                  return

             meta_path = possible_meta_paths[0]
             meta_blob = commit.tree[meta_path]
             metadata = json.loads(meta_blob.data_stream.read().decode('utf-8'))
             checksums = metadata.get("artifact_checksums", {})

             if checksums:
                 # Pretty print JSON checksums
                 checksum_str = json.dumps(checksums, indent=2, ensure_ascii=False)
                 self.ui.panel(checksum_str, f"저장된 체크섬 (Commit: {commit_hash})", border_style="magenta")
             else:
                  print(f"[dim]커밋 {commit_hash}에 저장된 아티팩트 체크섬 정보 없음.[/]")

         except GitCommandError:
             self.ui.error(f"커밋 해시 '{commit_hash}'를 찾을 수 없음.")
         except Exception as e:
             self.ui.error(f"체크섬 정보 로드/표시 중 오류 ({commit_hash})", traceback.format_exc())

# ─────────────── CLI Argument Parsing & Main ───────────────
def main():
    # Check if running inside a Git repository before parsing args maybe
    try:
        _ = Repo('.', search_parent_directories=True).working_tree_dir
        is_git_repo = True
    except InvalidGitRepositoryError:
        is_git_repo = False
        print("[bold yellow]경고: 현재 폴더는 Git 저장소가 아닙니다. 'save', 'load', 'diff', 'verify' 명령은 작동하지 않을 수 있습니다.[/]")
        # Allow continuing, maybe user only wants --help?

    parser = argparse.ArgumentParser(
        description="AI 기반 프로젝트 인수인계 상태 관리 도구 (v1.1)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("AI_BACKEND", "ollama"),
        choices=list(available_backends.keys()) if available_backends else ["none"], # Handle no backends found
        help=f"사용할 AI 백엔드 선택 (기본값: ollama 또는 AI_BACKEND env var). 사용 가능: {', '.join(available_backends.keys())}"
    )

    subparsers = parser.add_subparsers(dest="command", help="실행할 작업")
    subparsers.required = True # Make command mandatory

    # Save command
    parser_save = subparsers.add_parser("save", help="현재 작업 상태를 요약하여 저장 (Git 필요)")

    # Load command
    parser_load = subparsers.add_parser("load", help="과거 저장된 상태 불러오기 (Git 필요)")
    parser_load.add_argument("-l", "--latest", action="store_true", help="가장 최근 상태 자동 로드")

    # Diff command
    parser_diff = subparsers.add_parser("diff", help="현재 변경 사항 미리보기 (Git 필요)")
    parser_diff.add_argument("target", nargs="?", default="HEAD", help="비교 대상 Commit/Branch (기본값: HEAD)")

    # Verify command
    parser_verify = subparsers.add_parser("verify", help="저장된 상태 아티팩트 체크섬 표시 (Git 필요)")
    parser_verify.add_argument("commit", help="체크섬 정보를 확인할 상태 커밋 해시")

    args = parser.parse_args()

    # Exit if required command needs Git but we're not in a repo
    git_required_commands = ["save", "load", "diff", "verify"]
    if args.command in git_required_commands and not is_git_repo:
         UI.error(f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다.")
         sys.exit(1)

    if not available_backends and args.command != "--help": # Allow help even if no backend
        UI.error("사용 가능한 AI 백엔드가 없어 명령을 실행할 수 없습니다.")
        sys.exit(1)

    print(f"[bold underline]Handover 스크립트 v1.1[/]")

    try:
        # Initialize Handover only if a command requiring it is run
        # This avoids unnecessary Git/AI init if just showing help
        if args.command in git_required_commands: # Assuming all commands need Handover instance
            handler = Handover(backend_choice=args.backend)

            if args.command == "save":
                handler.save()
            elif args.command == "load":
                handler.load(latest=args.latest)
            elif args.command == "diff":
                handler.diff(target=args.target)
            elif args.command == "verify":
                handler.verify_checksums(commit_hash=args.commit)
        # else: Handle potential future commands not needing Handover instance?

    except Exception as e:
        # Catch errors during Handover init or command execution
        UI.error("스크립트 실행 중 예기치 않은 오류 발생", traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Add basic checks before running main maybe?
    if sys.version_info < (3, 8):
         print("[bold red]오류: 이 스크립트는 Python 3.8 이상 버전이 필요합니다.[/]")
         sys.exit(1)
    main()