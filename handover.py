#!/usr/bin/env python3
# handover.py – 인수인계 v1.1 (무료 · 로컬 LLM 전용 + 확장 기능)

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
from typing import List, Dict, Tuple, Optional, Type, Any # Added Any for config
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
    import markdown2
    # backends.base 는 아래에서 임포트
except ImportError as e:
    print(f"[bold red]오류: 필요한 라이브러리가 설치되지 않았습니다.[/]\n{e}")
    print("팁: [yellow]pip install gitpython requests rich python-dotenv markdown2[/] 명령을 실행하세요.")
    sys.exit(1)

# --- 환경 변수 로드 ---
load_dotenv()

# --- 경로 & 상수 ---
# ROOT 디렉토리 설정은 GitRepo 클래스 초기화 시점으로 이동하거나,
# 스크립트가 항상 Git repo 루트에서 실행된다고 가정.
# 우선 argparse 실행 후 GitRepo 초기화하도록 변경
ROOT_PATH_UNCHECKED = pathlib.Path('.').resolve() # argparse 실행 전 임시 경로

STATE_DIR = ROOT_PATH_UNCHECKED / "ai_states"
ART_DIR = ROOT_PATH_UNCHECKED / "artifacts"
BACKEND_DIR = pathlib.Path(__file__).resolve().parent / "backends" # 스크립트 기준
COMMIT_TAG = "state("

# 초기 디렉토리 생성은 main 함수 또는 Handover 클래스에서 수행하도록 변경
# STATE_DIR.mkdir(exist_ok=True)
# ART_DIR.mkdir(exist_ok=True)

# --- AI 백엔드 로딩 ---
# backends.base 모듈 임포트 시도
try:
    # backends 폴더가 스크립트와 같은 위치에 있다고 가정
    spec = importlib.util.spec_from_file_location("backends.base", BACKEND_DIR / "base.py")
    if spec is None or spec.loader is None:
        raise ImportError("backends.base 모듈 스펙을 찾을 수 없습니다.")
    backends_base_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backends_base_module)
    AIBaseBackend = backends_base_module.AIBaseBackend
except ImportError as e:
    print(f"[bold red]오류: backends.base 모듈 임포트 실패. 'backends/base.py' 파일 존재 및 경로 확인 필요: {e}[/]")
    print(f"BACKEND_DIR: {BACKEND_DIR}")
    sys.exit(1)
except AttributeError:
    print(f"[bold red]오류: backends.base 모듈에서 AIBaseBackend 클래스를 찾을 수 없습니다.[/]")
    sys.exit(1)


available_backends: Dict[str, Type[AIBaseBackend]] = {}
if BACKEND_DIR.exists() and BACKEND_DIR.is_dir():
    for f_py in BACKEND_DIR.glob("*.py"):
        module_name_stem = f_py.stem
        if module_name_stem == "__init__" or module_name_stem == "base":
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"backends.{module_name_stem}", f_py)
            if spec is None or spec.loader is None:
                print(f"[yellow]경고: 백엔드 모듈 스펙 로딩 실패 {f_py}[/]")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in module.__dict__.items():
                if (isinstance(obj, type) and
                        issubclass(obj, AIBaseBackend) and
                        obj is not AIBaseBackend):
                    backend_name_from_class = obj.get_name() # Ensure get_name is static or callable on class
                    if backend_name_from_class != "base": # Should be defined by actual backends
                        available_backends[backend_name_from_class] = obj
        except ImportError as e:
            print(f"[yellow]경고: 백엔드 모듈 로딩 실패 {f_py}: {e}[/]")
        except AttributeError as e: # e.g. get_name missing
             print(f"[yellow]경고: 백엔드 클래스 속성 오류 {f_py} ({name if 'name' in locals() else 'UnknownClass'}): {e}[/]")
        except Exception as e: # Catch any other unexpected errors during backend loading
            print(f"[yellow]경고: 백엔드 파일 {f_py} 처리 중 예외 발생: {e}[/]")
else:
    # This case should ideally not be hit if script setup is correct
    print(f"[yellow]경고: 백엔드 디렉토리 '{BACKEND_DIR}'를 찾을 수 없거나 디렉토리가 아닙니다.[/]")


if not available_backends:
    # This is a critical error if we proceed to use AIProvider
    # We might allow proceeding if a command doesn't need AI (e.g. only --help)
    # For now, assume most commands will need it.
    print("[bold yellow]경고: 사용 가능한 AI 백엔드가 없습니다. 'backends' 폴더와 파일을 확인하세요.[/]")
    # Do not sys.exit(1) here, let main() decide based on command

# --- AI Provider ---
class AIProvider:
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        if not available_backends: # Check again before instantiation
             raise RuntimeError("AIProvider 초기화 실패: 사용 가능한 AI 백엔드가 없습니다.")
        if backend_name not in available_backends:
            raise ValueError(f"알 수 없는 백엔드: {backend_name}. 사용 가능: {list(available_backends.keys())}")

        BackendClass = available_backends[backend_name]
        try:
            self.backend: AIBaseBackend = BackendClass(config)
            print(f"[dim]AI 백엔드 사용: [bold cyan]{backend_name}[/][/dim]")
        except Exception as e:
            print(f"[bold red]오류: 백엔드 '{backend_name}' 초기화 실패: {e}[/]")
            if hasattr(BackendClass, 'get_config_description'):
                print(f"[yellow]필요 설정:\n{BackendClass.get_config_description()}[/]")
            raise e

    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        return self.backend.make_summary(task, ctx, arts)

    def verify_summary(self, md: str) -> Tuple[bool, str]:
        is_ok, msg = self.backend.verify_summary(md)
        if is_ok: # Perform additional structural check if AI says OK
           lines = md.strip().split('\n')
           headers = [l.strip() for l in lines if l.startswith('#')]
           required_headers_structure = ["#", "## 목표", "## 진행", "## 결정", "## 결과", "## 다음할일", "## 산출물"]
           if len(headers) != len(required_headers_structure): # Strict check for exactly 7 headers
               is_ok = False
               msg = f"헤더 개수 불일치 (필수 {len(required_headers_structure)}개, 발견 {len(headers)}개)"
           else:
               for i, req_struct_start in enumerate(required_headers_structure):
                  if not headers[i].startswith(req_struct_start):
                      if i == 0 and headers[i].startswith("# "): # Allow title for first header
                          continue
                      is_ok = False
                      msg = f"헤더 #{i+1} 구조 오류: '{headers[i]}' (예상: '{req_struct_start}' 시작)"
                      break
        return is_ok, msg

    def load_report(self, md: str) -> str:
        return self.backend.load_report(md)

# --- GitRepo ---
class GitRepo:
    def __init__(self, repo_path: pathlib.Path):
        try:
            self.repo = Repo(repo_path)
            if self.repo.bare:
                 raise InvalidGitRepositoryError(f"'{repo_path}'는 bare 저장소입니다.")
        except InvalidGitRepositoryError:
            # This error is now handled in main() before GitRepo is instantiated for git-dependent commands
            raise # Re-raise for main to catch if it still happens

    def _safe(self, git_func, *args, **kwargs):
        try:
            return git_func(*args, **kwargs)
        except GitCommandError as e:
            stderr = e.stderr.strip()
            raise RuntimeError(f"Git 명령어 실패: {e.command}\n오류: {stderr}") from e

    def get_last_state_commit(self) -> Optional[Commit]:
         try:
             for commit in self.repo.iter_commits(max_count=200, first_parent=True):
                 if commit.message.startswith(COMMIT_TAG):
                     return commit
         except Exception as e:
             print(f"[yellow]경고: 마지막 상태 커밋 조회 실패: {e}[/]")
         return None

    def get_commit_messages_since(self, commit_hash: Optional[str]) -> str:
        if not self.repo: return "Git 저장소가 초기화되지 않았습니다."
        if not commit_hash:
            try:
                commits = list(self.repo.iter_commits(max_count=10, no_merges=True))
                log = "\n".join(f"- {c.hexsha[:7]}: {c.summary}" for c in reversed(commits))
                return f"최근 커밋 {len(commits)}개:\n{log}" if log else "최근 커밋 없음"
            except Exception as e:
                 print(f"[yellow]경고: 최근 커밋 로그 조회 실패: {e}[/]")
                 return "최근 커밋 로그를 가져올 수 없습니다."
        try:
            # Ensure commit_hash is valid before using in range
            self.repo.commit(commit_hash) # Will raise if not found
            log_cmd = f"{commit_hash}..HEAD"
            commit_log = self.repo.git.log(log_cmd, '--pretty=format:- %h: %s', '--abbrev-commit', '--no-merges')
            return f"'{commit_hash[:8]}' 이후 커밋:\n{commit_log}" if commit_log else f"'{commit_hash[:8]}' 이후 커밋 없음"
        except GitCommandError as e:
            print(f"[yellow]경고: 커밋 로그 조회 실패 ({commit_hash}): {e.stderr}[/]")
            return f"'{commit_hash[:8]}' 이후 커밋 로그를 가져올 수 없습니다."
        except Exception as e: # Catch other exceptions like commit not found
            print(f"[yellow]경고: 커밋 로그 조회 중 오류 ({commit_hash}): {e}[/]")
            return f"'{commit_hash[:8]}' 이후 커밋 로그를 가져올 수 없습니다."


    def get_current_branch(self) -> Optional[str]:
        if not self.repo: return "Git 저장소 없음"
        try:
            return self.repo.active_branch.name
        except TypeError: # Detached HEAD
            try:
                return f"DETACHED_HEAD@{self.repo.head.commit.hexsha[:7]}"
            except Exception:
                 return "DETACHED_HEAD" # Fallback
        except Exception as e:
            print(f"[yellow]경고: 현재 브랜치 이름 조회 실패: {e}[/]")
            return None

    def get_diff(self, target: str = "HEAD", color: bool = True) -> str:
        if not self.repo: return "Git 저장소가 초기화되지 않았습니다."
        try:
            # Validate target reference
            try:
                self.repo.commit(target) # Works for commits, tags, branches (resolves to commit)
            except (GitCommandError, ValueError) as e:
                raise ValueError(f"비교 대상 '{target}'을 찾을 수 없습니다: {e}") from e

            color_opt = '--color=always' if color else '--color=never'
            # Staged changes (index vs target)
            staged_diff = self.repo.git.diff('--staged', target, color_opt)
            # Unstaged changes in working directory (working tree vs index)
            unstaged_wt_vs_index_diff = self.repo.git.diff(color_opt)
            # For overall changes vs target, we often want working tree vs target
            working_tree_vs_target_diff = self.repo.git.diff(target, color_opt)


            diff_output = ""
            has_staged = bool(staged_diff.strip())
            has_unstaged_vs_target = bool(working_tree_vs_target_diff.strip())

            if has_staged:
                diff_output += f"--- Staged Changes (vs {target}) ---\n{staged_diff}\n\n"

            # Only show working tree vs target if it's different from staged vs target,
            # or if there are no staged changes but there are working tree changes.
            if has_unstaged_vs_target and working_tree_vs_target_diff != staged_diff:
                diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"
            elif not has_staged and has_unstaged_vs_target: # Only working tree changes
                diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"


            return diff_output.strip() if diff_output.strip() else f"'{target}'과(와) 변경 사항 없음 (Git 추적 파일 기준)"

        except GitCommandError as e:
            return f"Diff 생성 오류: {e.stderr}"
        except ValueError as e: # Catch target validation error
             return f"Diff 생성 오류: {e}"
        except Exception as e:
             return f"Diff 생성 중 알 수 없는 오류: {e}"

    def save(self, state_paths: List[pathlib.Path], task: str, snapshot_dir: Optional[pathlib.Path]) -> str:
        if not self.repo: raise RuntimeError("Git 저장소가 없어 저장할 수 없습니다.")

        paths_to_add_str = [str(p.resolve()) for p in state_paths]
        if snapshot_dir and snapshot_dir.exists() and any(snapshot_dir.iterdir()):
            paths_to_add_str.append(str(snapshot_dir.resolve()))

        if not state_paths: # Markdown and Meta are essential
             raise ValueError("저장할 상태 파일(.md, .meta.json)이 없습니다.")

        self._safe(self.repo.git.add, *paths_to_add_str)

        # Only commit if there are actual changes staged for commit
        # (excluding untracked files that were just added, like a new snapshot dir)
        # repo.is_dirty() checks working tree. We need to check index vs HEAD.
        if not self.repo.index.diff("HEAD") and not self.repo.untracked_files:
             # This check might be too strict if only untracked files were added and staged
             # A simpler check: if paths_to_add_str contains only new untracked files.
             # Let's check if `git commit` would do anything.
             # The `repo.index.commit` will fail if nothing to commit, or create empty if allow_empty.
             # We assume _safe handles this or we ensure commit only if diffs exist.
             # For simplicity, try to commit. Git itself might return non-zero if nothing changed.
             pass # Allow commit attempt

        commit_message = f"{COMMIT_TAG}{task})"
        # Commit, allow empty if user forces it (not implemented) or if untracked added
        try:
             self._safe(self.repo.index.commit, commit_message)
        except RuntimeError as e:
            if "nothing to commit" in str(e).lower() or "no changes added to commit" in str(e).lower():
                print("[yellow]경고: 커밋할 변경 사항이 없습니다. 이전 상태와 동일합니다.[/]")
                # Return current HEAD if no changes
                return self.repo.head.commit.hexsha[:8] + " (변경 없음)"
            raise e # Re-raise other commit errors

        # Push
        if self.repo.remotes:
            try:
                # Push current branch. Need to handle if upstream is not set.
                current_branch_name = self.get_current_branch()
                if current_branch_name and not current_branch_name.startswith("DETACHED_HEAD"):
                    # Simplistic push, might need more robust upstream handling
                    self._safe(self.repo.git.push, 'origin', current_branch_name)
                    print("[green]원격 저장소에 푸시 완료.[/]")
                else:
                    print(f"[yellow]경고: 현재 브랜치({current_branch_name})를 특정할 수 없어 푸시를 건너<0xEB><0x9B><0x8D>니다.[/]")
            except RuntimeError as e:
                print(f"[yellow]경고: 원격 저장소 푸시 실패. 로컬에는 커밋되었습니다. ({e})[/]")
        else:
            print("[yellow]경고: 설정된 원격 저장소가 없어 푸시를 건너<0xEB><0x9B><0x8D>니다.[/]")

        return self.repo.head.commit.hexsha[:8]

    def list_states(self) -> List[Dict]:
        if not self.repo: return [] # No repo, no states
        items = []
        try:
            commits = list(self.repo.iter_commits(max_count=100, first_parent=True, paths=str(STATE_DIR)))
        except Exception as e:
            # If STATE_DIR doesn't exist yet, this might fail. Fallback to all commits.
            try:
                commits = list(self.repo.iter_commits(max_count=100, first_parent=True))
            except Exception as e_all:
                raise RuntimeError(f"Git 커밋 목록 조회 중 오류: {e_all}") from e_all

        for commit in commits:
            if not commit.message.startswith(COMMIT_TAG): continue

            meta_blob: Optional[Blob] = None
            headline = ""
            meta_found_in_commit = False

            try:
                # Check if STATE_DIR itself is in the commit's tree
                # This assumes STATE_DIR is relative to repo root.
                if STATE_DIR.name not in commit.tree:
                    # If state dir itself is not in commit tree, skip (maybe an old commit structure)
                    # However, paths in iter_commits should have filtered this already if STATE_DIR exists.
                    # This logic might be redundant if iter_commits(paths=...) works as expected.
                    # Let's search for meta files within the commit tree more broadly if path filter fails.
                    pass


                for item in commit.tree[STATE_DIR.name].traverse(): # Traverse only within STATE_DIR
                    if isinstance(item, Blob) and item.name.endswith(".meta.json"):
                         meta_blob = item
                         meta_found_in_commit = True
                         break
                if not meta_blob and not meta_found_in_commit: # Try a broader search if specific path fails
                     for item in commit.tree.traverse():
                          if isinstance(item, Blob) and item.path.startswith(STATE_DIR.name) and item.path.endswith(".meta.json"):
                              meta_blob = item
                              break


                if meta_blob:
                    try:
                        metadata = json.loads(meta_blob.data_stream.read().decode('utf-8'))
                        headline = metadata.get("headline", "")
                    except Exception as e_parse:
                        print(f"[yellow]경고: 커밋 {commit.hexsha[:8]} 메타데이터 파싱 실패 ({meta_blob.path}, {e_parse})[/]")
                        headline = "[메타데이터 오류]"
            except KeyError: # STATE_DIR.name not in commit.tree
                 # This commit might not have the STATE_DIR, so no state files.
                 # This is fine if iter_commits(paths=...) already handles it.
                 # If iter_commits didn't use paths, this means it's not a state commit for our files.
                 # print(f"[dim]커밋 {commit.hexsha[:8]}에 {STATE_DIR.name} 없음 (메타데이터 스킵).[/]")
                 pass # No state dir, no meta for this commit
            except Exception as e_traverse:
                 print(f"[yellow]경고: 커밋 {commit.hexsha[:8]} 트리 탐색 중 오류 ({e_traverse})[/]")


            items.append({
                "hash": commit.hexsha[:8],
                "task": commit.message[len(COMMIT_TAG):-1].strip(),
                "time": datetime.datetime.fromtimestamp(commit.committed_date).strftime("%Y-%m-%d %H:%M"),
                "head": headline or "-"
            })

        return list(reversed(items))

    def load_state(self, commit_hash: str) -> str:
        if not self.repo: raise RuntimeError("Git 저장소가 없어 로드할 수 없습니다.")
        try:
            commit_obj = self.repo.commit(self.repo.git.rev_parse(commit_hash))
        except GitCommandError:
             raise RuntimeError(f"커밋 해시 '{commit_hash}'를 찾을 수 없습니다.") from None
        except Exception as e:
             raise RuntimeError(f"커밋 '{commit_hash}' 접근 중 오류: {e}") from e

        # Search for .md file within STATE_DIR of that commit
        try:
            state_dir_in_commit = commit_obj.tree[STATE_DIR.name]
            for item in state_dir_in_commit.traverse():
                if isinstance(item, Blob) and item.name.endswith(".md"):
                    try:
                        return item.data_stream.read().decode('utf-8')
                    except Exception as e_read:
                        raise RuntimeError(f"커밋 '{commit_hash}'에서 상태 파일 '{item.path}' 로드 실패: {e_read}") from e_read
            # If loop finishes, MD file not found in state_dir_in_commit
            # Try broader search as fallback (might be needed if STATE_DIR name changed over time)
            for item in commit_obj.tree.traverse():
                 if isinstance(item, Blob) and item.path.startswith(STATE_DIR.name) and item.path.endswith(".md"):
                     try:
                         return item.data_stream.read().decode('utf-8')
                     except Exception as e_read:
                        raise RuntimeError(f"커밋 '{commit_hash}'에서 상태 파일 '{item.path}' 로드 실패 (fallback): {e_read}") from e_read

            raise RuntimeError(f"커밋 '{commit_hash}'의 '{STATE_DIR.name}' 폴더에서 상태 파일(.md)을 찾을 수 없습니다.")

        except KeyError: # STATE_DIR.name not in commit_obj.tree
            raise RuntimeError(f"커밋 '{commit_hash}'에 '{STATE_DIR.name}' 폴더가 없습니다. 상태 파일 구조가 다를 수 있습니다.")
        except Exception as e:
            raise RuntimeError(f"커밋 '{commit_hash}' 상태 로드 중 예외 발생: {e}")


# --- Serializer ---
class Serializer:
    @staticmethod
    def _calculate_sha256(filepath: pathlib.Path) -> Optional[str]:
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while True:
                    byte_block = f.read(4096)
                    if not byte_block: break
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except IOError as e:
            print(f"[yellow]경고: 파일 해시 계산 실패 {filepath.name}: {e}[/]")
            return None

    @staticmethod
    def _generate_html(md_content: str, title: str) -> str:
         css = """<style>body{font-family:sans-serif;line-height:1.6;padding:20px;max-width:800px;margin:auto;color:#333}h1,h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.5em;margin-bottom:1em}h1{font-size:2em}h2{font-size:1.5em}ul,ol{padding-left:2em}li{margin-bottom:.5em}code{background-color:#f0f0f0;padding:.2em .4em;border-radius:3px;font-family:monospace;font-size:.9em}pre{background-color:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}pre code{background-color:transparent;padding:0;border-radius:0}blockquote{border-left:4px solid #ccc;padding-left:1em;color:#666;margin-left:0}table{border-collapse:collapse;width:100%;margin-bottom:1em}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}</style>"""
         html_body = markdown2.markdown(md_content, extras=["metadata", "fenced-code-blocks", "tables", "strike", "task_list", "code-friendly"])
         title_from_meta = title
         if hasattr(html_body, "metadata") and html_body.metadata.get("title"): title_from_meta = html_body.metadata["title"]
         return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{title_from_meta}</title>{css}</head><body>{html_body}</body></html>"""

    @staticmethod
    def save_state(md: str, task: str) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        safe_task_name = "".join(c for c in task if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        if not safe_task_name: safe_task_name = "untitled_task"
        base_filename = f"{ts}_{safe_task_name}"

        # Ensure STATE_DIR and ART_DIR exist before writing to them
        STATE_DIR.mkdir(exist_ok=True)
        ART_DIR.mkdir(exist_ok=True)

        state_file = STATE_DIR / f"{base_filename}.md"
        html_file = STATE_DIR / f"{base_filename}.html"
        meta_file = STATE_DIR / f"{base_filename}.meta.json"

        try: state_file.write_text(md, encoding="utf-8")
        except IOError as e: raise RuntimeError(f"상태 MD 파일 저장 실패 ({state_file}): {e}") from e

        html_saved_successfully = False
        try:
            html_content = Serializer._generate_html(md, task)
            html_file.write_text(html_content, encoding="utf-8")
            html_saved_successfully = True
        except Exception as e: print(f"[yellow]경고: HTML 프리뷰 생성/저장 실패 ({html_file.name}): {e}[/]")

        snapshot_dir: Optional[pathlib.Path] = None
        artifact_checksums: Dict[str, str] = {}
        artifact_files = [f for f in ART_DIR.iterdir() if f.is_file()] if ART_DIR.exists() else []

        if artifact_files:
            snapshot_dir_name = f"{base_filename}_artifacts"
            snapshot_dir = ART_DIR / snapshot_dir_name # Store artifacts inside ART_DIR
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            print(f"[dim]아티팩트 스냅샷 생성 및 체크섬 계산 중... ({len(artifact_files)}개 파일) -> '{snapshot_dir.relative_to(ROOT_PATH_UNCHECKED)}'[/]")
            for f in artifact_files:
                try:
                    target_path = snapshot_dir / f.name
                    shutil.copy2(f, target_path)
                    checksum = Serializer._calculate_sha256(target_path)
                    if checksum: artifact_checksums[f.name] = checksum
                except (IOError, shutil.Error) as copy_err: print(f"[yellow]경고: 아티팩트 파일 복사/해시 실패 ({f.name}): {copy_err}[/]")
            if artifact_checksums: print(f"[dim]체크섬 계산 완료 ({len(artifact_checksums)}개 파일).[/]")
        else: print("[dim]복사할 아티팩트 파일이 없습니다.[/]")

        headline = md.splitlines()[0].lstrip('# ').strip() if md.strip() and md.splitlines()[0].startswith("#") else task
        metadata = {"task": task, "ts": ts, "headline": headline, "artifact_checksums": artifact_checksums}
        try: meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except IOError as e: raise RuntimeError(f"메타데이터 파일 저장 실패 ({meta_file}): {e}") from e

        paths_to_add = [state_file, meta_file]
        if html_saved_successfully and html_file.exists(): paths_to_add.append(html_file)

        # Return snapshot_dir only if it was created and potentially has files
        # even if some checksums failed. Git will handle empty dir if all copies failed.
        return paths_to_add, snapshot_dir if snapshot_dir and snapshot_dir.exists() else None

    @staticmethod
    def to_prompt(md: str, commit: str) -> str:
        return f"### 이전 상태 정보 (Commit: {commit}) ###\n\n{md}\n\n### 상태 정보 끝 ###"

# --- UI ---
class UI:
    console = Console()
    @staticmethod
    def task_name(default: str = "작업 요약") -> str:
        return Prompt.ask("[bold cyan]작업 이름[/]", default=default)

    @staticmethod
    def multiline(label: str, default: str = "") -> str:
        UI.console.print(f"\n[bold cyan]{label}[/]")
        UI.console.print("[dim](입력을 마치려면 빈 줄에서 Enter 키를 두 번 누르세요)[/]")
        lines = []
        if default:
             print(Panel(default, title="[dim]자동 제안된 내용 (편집 가능)[/]", border_style="dim", expand=False))
             print("[dim]--------------------[/]")
             # Do not pre-fill lines, user types fresh or pastes.

        blank_count = 0
        while True:
            try: line = input()
            except EOFError: break
            if line == "":
                blank_count += 1
                if blank_count >= 2: break
            else:
                blank_count = 0; lines.append(line)
        final_text = "\n".join(lines).strip()
        if not final_text and default:
             print("[dim]입력이 없어 자동 제안된 내용으로 진행합니다.[/]")
             return default
        return final_text

    @staticmethod
    def notify(msg: str, style: str = "green"): UI.console.print(f"\n[bold {style}]✔ {msg}[/]")
    @staticmethod
    def error(msg: str, details: Optional[str] = None):
        UI.console.print(f"\n[bold red]❌ 오류: {msg}[/]")
        if details: UI.console.print(Panel(details, title="[dim]상세 정보[/]", border_style="dim red", expand=False))

    @staticmethod
    def pick_state(states: List[Dict]) -> Optional[str]:
        if not states: print("[yellow]저장된 상태가 없습니다.[/]"); return None
        tb = Table(title="[bold]저장된 인수인계 상태[/]",box=box.ROUNDED,show_lines=True)
        tb.add_column("#",style="dim",justify="right"); tb.add_column("커밋");tb.add_column("작업");tb.add_column("시각");tb.add_column("헤드라인", overflow="fold")
        for i,s in enumerate(states): tb.add_row(str(i),s["hash"],s["task"],s["time"],s["head"])
        UI.console.print(tb)
        choices = [str(i) for i in range(len(states))]
        sel = Prompt.ask("[bold cyan]로드할 상태 번호 (취소: Enter)[/]",choices=choices+[""],default="",show_choices=False)
        if sel.isdigit() and 0 <= int(sel) < len(states):
            sel_hash = states[int(sel)]["hash"]; print(f"[info]선택: {sel_hash} ({states[int(sel)]['task']})[/]"); return sel_hash
        print("[info]로드 취소.[/]"); return None

    @staticmethod
    def panel(txt: str, title: str, border_style: str = "blue"):
        UI.console.print(Panel(txt, title=f"[bold]{title}[/]", border_style=border_style, expand=False, padding=(1,2)))

    @staticmethod
    def diff_panel(diff_text: str, target: str):
        if not diff_text.strip() or "변경 사항 없음" in diff_text or "오류" in diff_text :
            print(f"[dim]{diff_text}[/]") # Print "no changes" or error plainly
            return
        syntax = Syntax(diff_text, "diff", theme="default", line_numbers=False, word_wrap=False)
        UI.console.print(Panel(syntax, title=f"[bold]Diff (vs {target})[/]", border_style="yellow", expand=True))

# --- Handover (Core Logic) ---
class Handover:
    def __init__(self, backend_choice: str, current_root: pathlib.Path):
        self.ui = UI()
        self.root_dir = current_root # Store the validated root_dir
        try:
             self.git = GitRepo(self.root_dir)
        except InvalidGitRepositoryError: # Handled by main, but as safeguard
             self.ui.error(f"'{self.root_dir}'는 Git 저장소가 아닙니다. 'save' 등 Git 관련 명령 사용 불가.")
             # sys.exit(1) # Don't exit, allow --help or non-git commands if any
             self.git = None # Mark as no git
        except Exception as e:
             self.ui.error(f"GitRepo 초기화 실패: {e}", traceback.format_exc())
             sys.exit(1) # Critical if GitRepo fails beyond not being a repo

        # Initialize AIProvider only if a backend is chosen and available
        if backend_choice != "none" and available_backends:
            try:
                self.ai = AIProvider(backend_name=backend_choice, config={})
            except Exception as e:
                 self.ui.error(f"AI 백엔드 ({backend_choice}) 초기화 실패.", traceback.format_exc())
                 sys.exit(1)
        elif backend_choice != "none" and not available_backends:
             self.ui.error(f"선택된 백엔드 '{backend_choice}'를 위한 모듈을 찾을 수 없거나 로드 중 오류 발생.")
             sys.exit(1)
        else: # backend_choice is "none"
            self.ai = None # No AI
            self.ui.console.print("[yellow]경고: AI 백엔드가 'none'으로 설정되어 AI 기능이 비활성화됩니다.[/]")


    def _ensure_git_ai(self, command_name: str):
        """Checks if Git and AI are available for commands that need them."""
        if not self.git:
            self.ui.error(f"'{command_name}' 명령은 Git 저장소 내에서 실행해야 합니다.")
            sys.exit(1)
        if not self.ai:
            self.ui.error(f"'{command_name}' 명령은 AI 백엔드가 설정되어야 합니다. (--backend 옵션 확인)")
            sys.exit(1)

    def save(self):
        self._ensure_git_ai("save")
        try:
            default_task = self.git.get_current_branch() or self.git.repo.head.commit.summary
            task = self.ui.task_name(default=default_task)

            last_state_commit = self.git.get_last_state_commit()
            default_context = self.git.get_commit_messages_since(last_state_commit.hexsha if last_state_commit else None)
            context = self.ui.multiline("작업 내용 요약 (AI 요약 결과 붙여넣기 권장)", default=default_context)
            if not context.strip(): self.ui.error("작업 내용 요약이 비어있어 저장을 취소합니다."); return

            artifacts = [f.name for f in ART_DIR.iterdir() if f.is_file()] if ART_DIR.exists() else []
            if ART_DIR.exists(): self.ui.console.print(f"[dim]현재 아티팩트 ({ART_DIR.relative_to(self.root_dir)}): {', '.join(artifacts) or '없음'}[/]")
            else: self.ui.console.print(f"[dim]아티팩트 폴더({ART_DIR.relative_to(self.root_dir)}) 없음.[/]")


            self.ui.console.print("\n[bold yellow]AI가 인수인계 문서를 생성 중입니다...[/]")
            markdown_summary = self.ai.make_summary(task, context, artifacts)
            self.ui.panel(markdown_summary, "AI 생성 요약본 (검증 전)")

            self.ui.console.print("[bold yellow]생성된 요약본을 AI가 검증 중입니다...[/]")
            is_ok, validation_msg = self.ai.verify_summary(markdown_summary)
            if not is_ok: raise RuntimeError(f"AI 생성 문서 검증 실패:\n{validation_msg}")
            self.ui.notify("AI 검증 통과!", style="green")

            # Ensure STATE_DIR and ART_DIR are using the correct ROOT
            global STATE_DIR, ART_DIR # Allow modification if root_dir changed
            STATE_DIR = self.root_dir / "ai_states"
            ART_DIR = self.root_dir / "artifacts"

            state_files, snapshot_dir = Serializer.save_state(markdown_summary, task)
            commit_hash = self.git.save(state_files, task, snapshot_dir)
            self.ui.notify(f"인수인계 상태 저장 완료! (Commit: {commit_hash})", style="bold green")

            html_file = next((f for f in state_files if f.name.endswith(".html")), None)
            if html_file and html_file.exists():
                 self.ui.console.print(f"[dim]HTML 프리뷰: {html_file.relative_to(self.root_dir)}[/]")

        except Exception as e: self.ui.error(f"Save 중 오류: {str(e)}", traceback.format_exc())

    def load(self, latest: bool = False):
        self._ensure_git_ai("load")
        try:
            saved_states = self.git.list_states()
            if not saved_states: self.ui.error("저장된 상태가 없습니다."); return
            selected_hash = saved_states[-1]["hash"] if latest else self.ui.pick_state(saved_states)
            if not selected_hash: return

            if latest: print(f"[info]최신 상태 로드 중: {selected_hash} ({next((s['task'] for s in saved_states if s['hash']==selected_hash),'')})[/]")

            self.ui.console.print(f"[bold yellow]{selected_hash} 커밋에서 상태 정보 로드 중...[/]")
            markdown_content = self.git.load_state(selected_hash)
            prompt_format = Serializer.to_prompt(markdown_content, selected_hash)
            self.ui.panel(prompt_format, f"로드된 상태 (Commit: {selected_hash})", border_style="cyan")

            self.ui.console.print("[bold yellow]AI가 로드된 상태를 분석/요약합니다...[/]")
            report = self.ai.load_report(markdown_content)
            self.ui.panel(report, "AI 이해도 보고서", border_style="magenta")
        except Exception as e: self.ui.error(f"Load 중 오류: {str(e)}", traceback.format_exc())

    def diff(self, target: str = "HEAD"):
        if not self.git: self.ui.error("'diff' 명령은 Git 저장소 내에서 실행해야 합니다."); sys.exit(1)
        try:
             self.ui.console.print(f"[bold yellow]'{target}' 대비 변경 사항 확인 중... (Git 추적 파일 기준)[/]")
             diff_output = self.git.get_diff(target, color=True)
             self.ui.diff_panel(diff_output, target)
        except Exception as e: self.ui.error(f"Diff 중 오류: {str(e)}", traceback.format_exc())

    def verify_checksums(self, commit_hash: str):
        if not self.git: self.ui.error("'verify' 명령은 Git 저장소 내에서 실행해야 합니다."); sys.exit(1)
        self.ui.console.print(f"[yellow]기능 구현 예정:[/][dim] 커밋 {commit_hash}의 아티팩트 체크섬 실제 파일과 비교 검증[/]")
        self.ui.console.print("[dim](현재는 저장 시 기록된 체크섬 정보만 보여줍니다)[/]")
        try:
            commit = self.git.repo.commit(self.git.repo.git.rev_parse(commit_hash)) # Validate commit
            global STATE_DIR # Use correct STATE_DIR relative to this repo
            STATE_DIR = self.root_dir / "ai_states"

            meta_blob = None
            # Search for meta file in the commit's STATE_DIR
            # This assumes STATE_DIR name hasn't changed and is at root of commit content
            if STATE_DIR.name in commit.tree:
                for item in commit.tree[STATE_DIR.name].traverse():
                    if isinstance(item, Blob) and item.name.endswith(".meta.json"):
                        meta_blob = item; break
            if not meta_blob: # Broader search if not found in expected dir structure
                 for item in commit.tree.traverse():
                      if isinstance(item, Blob) and item.path.startswith(STATE_DIR.name) and item.path.endswith(".meta.json"):
                          meta_blob = item; break


            if not meta_blob: self.ui.error(f"커밋 {commit_hash}에서 메타데이터 파일(.meta.json)을 찾을 수 없음."); return

            metadata = json.loads(meta_blob.data_stream.read().decode('utf-8'))
            checksums = metadata.get("artifact_checksums", {})
            if checksums:
                self.ui.panel(json.dumps(checksums,indent=2,ensure_ascii=False), f"저장된 체크섬 (Commit: {commit_hash})", border_style="magenta")
            else: print(f"[dim]커밋 {commit_hash}에 저장된 아티팩트 체크섬 정보 없음.[/]")
        except GitCommandError: self.ui.error(f"커밋 해시 '{commit_hash}'를 찾을 수 없음.")
        except Exception as e: self.ui.error(f"체크섬 정보 로드 중 오류 ({commit_hash})", traceback.format_exc())

# --- CLI Argument Parsing & Main ---
def main():
    global ROOT_PATH_UNCHECKED, STATE_DIR, ART_DIR # Allow main to set validated ROOT

    is_git_repo = False
    current_git_root = None
    try:
        current_git_root = pathlib.Path(Repo('.', search_parent_directories=True).working_tree_dir)
        is_git_repo = True
        ROOT_PATH_UNCHECKED = current_git_root # Update with actual Git root
    except InvalidGitRepositoryError:
        # Keep ROOT_PATH_UNCHECKED as Path('.') if not in a git repo
        pass # Handled by command checks later

    # Update STATE_DIR and ART_DIR based on potentially found Git root
    STATE_DIR = ROOT_PATH_UNCHECKED / "ai_states"
    ART_DIR = ROOT_PATH_UNCHECKED / "artifacts"


    parser = argparse.ArgumentParser(description="AI 기반 프로젝트 인수인계 상태 관리 도구 (v1.1.1)", formatter_class=argparse.RawTextHelpFormatter)
    backend_choices = list(available_backends.keys()) if available_backends else []
    parser.add_argument("--backend", default=os.getenv("AI_BACKEND", "ollama" if "ollama" in backend_choices else (backend_choices[0] if backend_choices else "none")),
                        choices=backend_choices + ["none"], # Add "none" as a valid choice
                        help=f"사용할 AI 백엔드 (기본값: ollama 또는 첫번째 감지된 백엔드, AI_BACKEND env var). 사용 가능: {', '.join(backend_choices) or '없음'}. 'none'으로 비활성화.")

    subparsers = parser.add_subparsers(dest="command", help="실행할 작업")
    subparsers.required = True

    for cmd_name, help_text, git_needed, ai_needed in [
        ("save", "현재 작업 상태를 요약하여 저장", True, True),
        ("load", "과거 저장된 상태 불러오기", True, True),
        ("diff", "현재 변경 사항 미리보기 (Git)", True, False), # Diff doesn't need AI
        ("verify", "저장된 상태 아티팩트 체크섬 표시", True, False) # Verify doesn't need AI
    ]:
        p = subparsers.add_parser(cmd_name, help=f"{help_text}{' (Git 필요)' if git_needed else ''}{' (AI 필요)' if ai_needed else ''}")
        if cmd_name == "load": p.add_argument("-l", "--latest", action="store_true", help="가장 최근 상태 자동 로드")
        if cmd_name == "diff": p.add_argument("target", nargs="?", default="HEAD", help="비교 대상 Commit/Branch (기본값: HEAD)")
        if cmd_name == "verify": p.add_argument("commit", help="체크섬 정보를 확인할 상태 커밋 해시")

    args = parser.parse_args()

    # Ensure directories exist (now that ROOT_PATH_UNCHECKED is set based on git or cwd)
    # These are only strictly needed for 'save', but good to have for other commands if they evolve.
    STATE_DIR.mkdir(exist_ok=True)
    ART_DIR.mkdir(exist_ok=True)


    # Command-specific prerequisite checks
    command_needs_git = args.command in ["save", "load", "diff", "verify"]
    command_needs_ai = args.command in ["save", "load"] # Only save and load need AI Provider

    if command_needs_git and not is_git_repo:
        UI.error(f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 위치: {ROOT_PATH_UNCHECKED})")
        sys.exit(1)
    
    # If AI is needed but no backends or 'none' is chosen, block.
    if command_needs_ai and (not available_backends or args.backend == "none"):
        if not available_backends:
            UI.error(f"'{args.command}' 명령 실행 불가: 사용 가능한 AI 백엔드가 없습니다. 'backends' 폴더를 확인하세요.")
        else: # args.backend == "none"
            UI.error(f"'{args.command}' 명령 실행 불가: AI 백엔드가 'none'으로 설정되었습니다. --backend 옵션으로 AI 백엔드를 지정하세요.")
        sys.exit(1)


    print(f"[bold underline]Handover 스크립트 v1.1.1[/]")
    if is_git_repo and current_git_root:
         print(f"[dim]프로젝트 루트: {current_git_root}[/]")
    else:
         print(f"[dim]현재 작업 폴더: {ROOT_PATH_UNCHECKED} (Git 저장소 아님)[/]")


    try:
        # Instantiate Handover only if a command requiring it is run
        # And pass the validated current_git_root (or ROOT_PATH_UNCHECKED if not git repo)
        # This implies Handover class itself needs to handle if self.git is None for some commands.
        # For now, commands that need git check `is_git_repo` above.
        handler = Handover(backend_choice=args.backend, current_root=ROOT_PATH_UNCHECKED)

        if args.command == "save": handler.save()
        elif args.command == "load": handler.load(latest=args.latest)
        elif args.command == "diff": handler.diff(target=args.target)
        elif args.command == "verify": handler.verify_checksums(commit_hash=args.commit)

    except Exception as e:
        UI.error("스크립트 메인 실행 중 예기치 않은 오류", traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    if sys.version_info < (3, 8): # Basic Python version check
         print("[bold red]오류: 이 스크립트는 Python 3.8 이상 버전이 필요합니다.[/]")
         sys.exit(1)
    main()