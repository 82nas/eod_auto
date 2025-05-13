#!/usr/bin/env python3
# handover.py – 인수인계 v1.2.0 (수동 작성 모드)

from __future__ import annotations
import os
import sys
import datetime
import json
import textwrap # HTML 제목 단축 위해 필요
# from textwrap import dedent # dedent는 SYSTEM_INSTRUCTION_MD 제거로 불필요
import pathlib
import shutil
import argparse
import hashlib
# import importlib # AI 백엔드 로딩 제거
# import importlib.util # AI 백엔드 로딩 제거
import subprocess
import re
import traceback
from typing import List, Dict, Tuple, Optional, Type, Any # Type 제거 가능성 있음
from dotenv import load_dotenv

# --- 의존성 로드 ---
try:
    from git import Repo, GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Blob, Commit, HookExecutionError
    from rich import print, box
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.console import Console
    from rich.syntax import Syntax
    import markdown2 # type: ignore
except ImportError as e:
    print(f"[bold red]오류: 필요한 라이브러리가 설치되지 않았습니다.[/]\n{e}")
    print("팁: [yellow]pip install gitpython rich python-dotenv markdown2[/] 명령을 실행하세요.")
    sys.exit(1)

load_dotenv()

# BACKEND_DIR 제거
COMMIT_TAG = "state("
# SYSTEM_INSTRUCTION_MD 제거

# AI 백엔드 로딩 코드 전체 제거

# AIProvider 클래스 전체 제거

# --- GitRepo 클래스 (변경 없음) ---
class GitRepo:
    def __init__(self, repo_path: pathlib.Path):
        self.repo: Optional[Repo] = None
        self.repo_root_path: Optional[pathlib.Path] = None
        try:
            self.repo = Repo(str(repo_path), search_parent_directories=True)
            self.repo_root_path = pathlib.Path(self.repo.working_dir)
            if self.repo_root_path.resolve() != repo_path.resolve() and repo_path.resolve() in self.repo_root_path.parents:
                print(f"[dim]Git 저장소 루트를 '{self.repo_root_path}'로 감지했습니다.[/dim]")
            if self.repo.bare:
                print(f"[yellow]경고: '{self.repo_root_path}'는 bare 저장소입니다. 일부 기능이 제한될 수 있습니다.[/]")
        except InvalidGitRepositoryError:
            print(f"[yellow]경고: '{repo_path}' 또는 상위 디렉토리에 유효한 Git 저장소가 없습니다. Git 관련 기능 비활성화됨.[/yellow]")
        except NoSuchPathError:
            print(f"[red]오류: Git 저장소 경로를 찾을 수 없습니다: {repo_path}[/]")
        except GitCommandError as e:
            print(f"[red]오류: Git 저장소 초기화 중 Git 명령어 실행 실패 ({e.command}). Git 설치 및 PATH 확인 필요.[/]")
            print(f"[dim]상세: {e.stderr}[/dim]")
        except Exception as e:
            print(f"[red]오류: Git 저장소 초기화 중 예기치 않은 오류 발생: {e}[/]")

    def _run_git_command(self, cmd_parts: List[str], check: bool = True, repo_path_override: Optional[str] = None) -> str:
        effective_repo_path = repo_path_override or (str(self.repo_root_path) if self.repo_root_path else None)
        if not effective_repo_path:
            print("[yellow]경고: Git 저장소 경로가 명확하지 않아 현재 디렉토리에서 Git 명령을 실행합니다.[/yellow]")
            effective_repo_path = "."
        try:
            process = subprocess.run(
                ["git"] + cmd_parts, cwd=effective_repo_path, capture_output=True, text=True,
                check=check, encoding='utf-8', errors='ignore'
            )
            return process.stdout.strip()
        except FileNotFoundError:
            raise RuntimeError("Git 명령어를 찾을 수 없습니다. Git이 설치되어 있고 시스템 PATH에 등록되어 있는지 확인하세요.")
        except subprocess.CalledProcessError as e:
            error_message = f"Git 명령어 실행 실패: git {' '.join(cmd_parts)}\n오류 코드: {e.returncode}\n"
            if e.stdout: error_message += f"STDOUT: {e.stdout.strip()}\n"
            if e.stderr: error_message += f"STDERR: {e.stderr.strip()}\n"
            raise RuntimeError(error_message)
        except Exception as e:
            raise RuntimeError(f"Git 명령어 'git {' '.join(cmd_parts)}' 실행 중 예기치 않은 오류 발생: {e}")

    def _parse_git_log(self, num_commits: int) -> List[Dict[str, str]]:
        # 이 메서드는 AI 컨텍스트용이었으나, 사용자 참고용으로 남겨둘 수 있음 (Handover.save에서 사용 여부 결정)
        if not self.repo: return []
        commits_data = []
        try:
            for commit in self.repo.iter_commits(max_count=num_commits):
                commits_data.append({
                    "hash": commit.hexsha,
                    "author": commit.author.name if commit.author else "N/A",
                    "date": datetime.datetime.fromtimestamp(commit.authored_date).strftime("%Y-%m-%d"),
                    "subject": commit.summary
                })
        except GitCommandError as e:
            print(f"[yellow]경고: Git 로그 파싱 중 Git 명령어 오류 발생 (GitPython): {e.stderr}[/yellow]")
            return []
        except Exception as e:
            print(f"[yellow]경고: Git 로그 파싱 중 예기치 않은 오류 발생 (GitPython): {e}[/yellow]")
            return []
        return commits_data # 최신순 로그 반환

    def _get_diff_summary_for_commit(self, commit_hash: str) -> Dict[str, Any]:
        # 이 메서드도 AI 컨텍스트용이었으나, 사용자 참고용으로 남겨둘 수 있음
        if not self.repo: return {"files": [], "raw_stat_summary": "Git 저장소 없음"}
        files_changed_summary = []
        raw_stat_output = ""
        try:
            commit_obj = self.repo.commit(commit_hash)
            if commit_obj.parents:
                stat_output = self._run_git_command(["show", "--stat=150,100", "--pretty=format:", commit_hash])
                raw_stat_output = stat_output[:2000]
                for line in stat_output.splitlines():
                    line_stripped = line.strip()
                    if not line_stripped or "file changed" in line_stripped.lower() or "files changed" in line_stripped.lower():
                        continue
                    match = re.match(r"(.+?)\s+\|\s+(\d+)\s*([+\-]{0,10})?", line_stripped)
                    if match:
                        file_path = match.group(1).strip()
                        try: changed_lines = int(match.group(2))
                        except ValueError: changed_lines = 0
                        plus_minus_dots = match.group(3) if match.group(3) else ""
                        insertions = plus_minus_dots.count('+')
                        deletions = plus_minus_dots.count('-')
                        files_changed_summary.append({
                            "file": file_path, "changed_lines": changed_lines,
                            "insertions": insertions, "deletions": deletions
                        })
            else: # 최초 커밋
                for file_path, stats in commit_obj.stats.files.items():
                    files_changed_summary.append({
                        "file": file_path, "changed_lines": stats.get('lines', 0),
                        "insertions": stats.get('insertions', 0), "deletions": stats.get('deletions', 0)
                    })
                raw_stat_output = "최초 커밋 (모든 파일 추가)"
        except Exception as e:
            print(f"[yellow]경고: 커밋 {commit_hash[:7]}의 diff 요약 생성 중 오류: {e}[/yellow]")
            return {"files": [], "raw_stat_summary": f"오류 발생: {str(e)[:100]}"}
        return {"files": files_changed_summary, "raw_stat_summary": raw_stat_output}

    def collect_recent_commits_info(self, num_commits: int = 10) -> List[Dict[str, Any]]:
        # 이 메서드도 AI 컨텍스트용이었으나, 사용자 참고용으로 남겨둘 수 있음
        if not self.repo:
            print("[yellow]Git 저장소가 초기화되지 않아 커밋 정보를 수집할 수 없습니다.[/yellow]")
            return []
        parsed_commits = self._parse_git_log(num_commits)
        if not parsed_commits: return []
        collected_data = []
        for commit_meta in parsed_commits:
            diff_info = self._get_diff_summary_for_commit(commit_meta["hash"])
            commit_meta_copy = commit_meta.copy()
            commit_meta_copy["changes"] = diff_info
            collected_data.append(commit_meta_copy)
        # 최신순으로 반환됨 (parsed_commits가 최신순)
        return collected_data

    def get_current_branch_name(self) -> Optional[str]:
        # 변경 없음
        if not self.repo: return None
        try:
            if self.repo.head.is_detached:
                return f"DETACHED_HEAD@{self.repo.head.commit.hexsha[:7]}"
            return self.repo.active_branch.name
        except Exception as e:
            print(f"[yellow]경고: 현재 브랜치 이름 가져오기 실패: {e}[/yellow]")
            return None

    def get_last_state_commit(self) -> Optional[Commit]:
         # 변경 없음
        if not self.repo: return None
        try:
            for c in self.repo.iter_commits(max_count=200, first_parent=True):
                if c.message.startswith(COMMIT_TAG): return c
        except Exception as e:
            print(f"[yellow]경고: 마지막 상태 커밋 검색 중 오류: {e}[/yellow]")
        return None

    def get_commit_messages_since(self, commit_hash: Optional[str]) -> str:
         # 변경 없음
        if not self.repo: return "Git 저장소 없음."
        if not commit_hash:
            try:
                commits = list(reversed(list(self.repo.iter_commits(max_count=10, no_merges=True))))
                log = "\n".join(f"- {c.hexsha[:7]}: {c.summary}" for c in commits)
                return f"최근 커밋 {len(commits)}개:\n{log}" if log else "최근 커밋 없음"
            except Exception as e: return f"최근 커밋 로그 조회 실패: {e}"
        try:
            log_items = []
            for commit in self.repo.iter_commits(rev=f"{commit_hash}..HEAD", no_merges=True):
                log_items.append(f"- {commit.hexsha[:7]}: {commit.summary}")
            commit_log = "\n".join(reversed(log_items))
            return f"'{commit_hash[:8]}' 이후 커밋:\n{commit_log}" if commit_log else f"'{commit_hash[:8]}' 이후 커밋 없음"
        except Exception as e:
            return f"커밋 로그 조회 중 오류 ({commit_hash}): {e}"

    def get_current_branch(self) -> Optional[str]:
        return self.get_current_branch_name()

    def get_diff(self, target: str = "HEAD", color: bool = True) -> str:
         # 변경 없음
        if not self.repo: return "Git 저장소 없음."
        try:
            staged_diff_text = self.repo.git.diff('--staged', target, color='always' if color else 'never')
            unstaged_diff_text = self.repo.git.diff(color='always' if color else 'never')
            diff_output = ""
            if staged_diff_text.strip():
                diff_output += f"--- Staged Changes (vs {target}) ---\n{staged_diff_text}\n\n"
            if unstaged_diff_text.strip():
                diff_output += f"--- Unstaged Changes in Working Directory (vs Index) ---\n{unstaged_diff_text}\n"
            return diff_output.strip() if diff_output.strip() else f"'{target}'과(와) 또는 Index와 변경 사항 없음."
        except GitCommandError as e: return f"Diff 생성 Git 오류: {e.stderr}"
        except Exception as e: return f"Diff 생성 오류: {e}"

    def save(self, state_paths: List[pathlib.Path], task: str, snapshot_dir: Optional[pathlib.Path]) -> str:
         # 변경 없음 (Git add, commit, push 로직 동일)
        if not self.repo: raise RuntimeError("Git 저장소가 없어 저장할 수 없습니다.")
        paths_to_add_abs = []
        repo_root = pathlib.Path(self.repo.working_dir)
        for p in state_paths:
            abs_p = p.resolve()
            try:
                abs_p.relative_to(repo_root)
                paths_to_add_abs.append(str(abs_p))
            except ValueError:
                print(f"[yellow]경고: Git 저장소 외부 경로 추가 시도 무시됨: {p}[/yellow]")

        if snapshot_dir and snapshot_dir.exists() and any(snapshot_dir.iterdir()):
            abs_snap_dir = snapshot_dir.resolve()
            try:
                abs_snap_dir.relative_to(repo_root)
                paths_to_add_abs.append(str(abs_snap_dir))
            except ValueError:
                 print(f"[yellow]경고: Git 저장소 외부 스냅샷 경로 추가 시도 무시됨: {snapshot_dir}[/yellow]")


        if not paths_to_add_abs: raise ValueError("저장할 상태 파일 또는 스냅샷 파일이 없습니다 (Git 저장소 내).")

        try:
             # add 전에 상태 확인 불필요 (add 후 커밋 시 Git이 판단)
             self.repo.index.add(paths_to_add_abs)
        except Exception as e_add: raise RuntimeError(f"Git add 작업 실패: {e_add}")

        commit_msg = f"{COMMIT_TAG}{task})"
        commit_hash = ""
        try:
            # 커밋 시도
            commit_obj = self.repo.index.commit(commit_msg)
            commit_hash = commit_obj.hexsha[:8]

        except HookExecutionError as e_hook: # Hook 오류 처리
            stderr_msg = str(e_hook.stderr) if hasattr(e_hook, 'stderr') else str(e_hook)
            wsl_error_match = re.search(r"execvpe\(/bin/bash\) failed: No such file or directory", stderr_msg)
            detailed_error = f"Git pre-commit 훅 실패 (WSL bash 실행 오류): {stderr_msg}" if wsl_error_match else f"Git pre-commit 훅 실패 (종료 코드: {e_hook.status}): {stderr_msg}"
            raise RuntimeError(detailed_error) from e_hook
        except GitCommandError as e_commit: # Git 명령 자체 오류 (stderr 확인 방식 유지)
            is_nothing_to_commit = False
            if hasattr(e_commit, 'stderr') and isinstance(getattr(e_commit, 'stderr'), str):
                 # stderr 내용으로 "nothing to commit" 확인 (불안정할 수 있음)
                if "nothing to commit" in getattr(e_commit, 'stderr').lower() or \
                   "no changes added to commit" in getattr(e_commit, 'stderr').lower():
                    is_nothing_to_commit = True

            if is_nothing_to_commit:
                print("[yellow]경고: 커밋할 변경 사항이 없습니다 (Git 예외 발생). 이전 상태와 동일할 수 있습니다.[/yellow]")
                commit_hash = self.repo.head.commit.hexsha[:8] + " (변경 없음)"
            else: # 그 외 GitCommandError
                raise RuntimeError(f"Git 커밋 중 오류 발생: {e_commit}") from e_commit
        except Exception as e_commit_other: # 기타 예외
            # "nothing to commit" 상태가 예외를 발생시키지 않는 경우도 고려 (add 후 상태 비교 등 필요 시 추가)
             raise RuntimeError(f"Git 커밋 중 예기치 않은 오류 발생: {e_commit_other}") from e_commit_other


        if self.repo.remotes: # Push 로직
            try:
                current_branch_name = self.get_current_branch_name()
                if current_branch_name and not current_branch_name.startswith("DETACHED_HEAD"):
                    print(f"[dim]'{current_branch_name}' 브랜치를 원격 저장소(origin)에 푸시 시도...[/]")
                    self.repo.git.push('origin', current_branch_name)
                    print("[green]원격 저장소에 푸시 완료.[/]")
                else:
                    print(f"[yellow]경고: 현재 브랜치({current_branch_name})가 특정되지 않았거나 Detached HEAD 상태이므로 푸시를 건너<0xEB><0x9A><0x81>니다.[/]")
            except GitCommandError as e_push:
                print(f"[yellow]경고: 원격 저장소 푸시 실패. 로컬에는 커밋되었습니다. ({e_push.stderr})[/]")
            except Exception as e_general:
                print(f"[yellow]경고: 원격 저장소 푸시 중 예기치 않은 오류: {e_general}[/]")
        else:
            print("[yellow]경고: 설정된 원격 저장소가 없어 푸시를 건너<0xEB><0x9A><0x81>니다.[/]")

        return commit_hash

    def _get_relative_path_str(self, target_path: pathlib.Path) -> Optional[str]:
         # 변경 없음
        if not self.repo_root_path: return None
        try:
            return target_path.resolve().relative_to(self.repo_root_path).as_posix()
        except ValueError: return None
        except Exception: return None

    def list_states(self, current_app_state_dir: pathlib.Path) -> List[Dict]:
         # 변경 없음 (최신순 반환)
        if not self.repo: return []
        items = []
        search_rel_path_str = self._get_relative_path_str(current_app_state_dir)
        if not search_rel_path_str:
            print(f"[yellow]경고: 상태 디렉토리({current_app_state_dir})가 Git 저장소 내에 없어 상태 목록을 검색할 수 없습니다.[/]")
            return []
        try:
            commits = list(self.repo.iter_commits(max_count=100, first_parent=True, paths=search_rel_path_str))
        except Exception as e:
            print(f"[yellow]경고: 특정 경로({search_rel_path_str}) 커밋 검색 실패: {e}[/]")
            return []

        for c in commits:
            if not c.message.startswith(COMMIT_TAG): continue
            headline = ""; meta_blob = None
            try:
                for item_in_tree in c.tree.traverse():
                    if isinstance(item_in_tree, Blob) and \
                       item_in_tree.path.startswith(search_rel_path_str) and \
                       item_in_tree.path.endswith(".meta.json"):
                        meta_blob = item_in_tree
                        break
                if meta_blob:
                    metadata = json.loads(meta_blob.data_stream.read().decode('utf-8'))
                    headline = metadata.get("headline", "")
            except Exception as e: headline = f"[메타데이터 오류: {e}]"

            items.append({
                "hash": c.hexsha[:8],
                "task": c.message[len(COMMIT_TAG):-1].strip(),
                "time": datetime.datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d %H:%M"),
                "head": headline or "-"
            })
        return items # 최신 커밋이 앞에 오도록 반환

    def load_state(self, commit_hash: str, current_app_state_dir: pathlib.Path) -> str:
         # 변경 없음
        if not self.repo: raise RuntimeError("Git 저장소가 없어 로드할 수 없습니다.")
        try: commit_obj = self.repo.commit(commit_hash)
        except Exception as e: raise RuntimeError(f"커밋 '{commit_hash}' 접근 오류: {e}") from e

        search_rel_path_str = self._get_relative_path_str(current_app_state_dir)
        if not search_rel_path_str:
            raise RuntimeError(f"상태 디렉토리({current_app_state_dir})가 Git 저장소 내에 없습니다.")

        try:
            for item_in_tree in commit_obj.tree.traverse():
                if isinstance(item_in_tree, Blob) and \
                   item_in_tree.path.startswith(search_rel_path_str) and \
                   item_in_tree.path.endswith(".md"):
                    return item_in_tree.data_stream.read().decode('utf-8')
            raise RuntimeError(f"커밋 '{commit_hash}' 내 경로 '{search_rel_path_str}'에서 상태 파일(.md)을 찾을 수 없습니다.")
        except Exception as e:
             if "에서 상태 파일(.md)을 찾을 수 없습니다" not in str(e):
                  raise RuntimeError(f"커밋 '{commit_hash}' 상태 로드 중 예기치 않은 오류: {e}") from e
             else:
                  raise e

# --- Serializer 클래스 (HTML 제목 생성 수정) ---
class Serializer:
    @staticmethod
    def _calculate_sha256(fp: pathlib.Path) -> Optional[str]:
        # 변경 없음
        h = hashlib.sha256()
        try:
            with open(fp, "rb") as f:
                while True:
                    b = f.read(4096)
                    if not b: break
                    h.update(b)
            return h.hexdigest()
        except IOError: return None
        except Exception: return None

    @staticmethod
    def _extract_headline(md: str, default_title: str) -> str:
        """Markdown 텍스트에서 첫 번째 레벨 1 또는 2 헤더를 추출"""
        headline = default_title # 기본값은 작업 이름
        for line in md.splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith("# "):
                headline = stripped_line.lstrip('# ').strip()
                break
            elif stripped_line.startswith("## "): # 레벨 2 헤더도 고려
                 headline = stripped_line.lstrip('## ').strip()
                 break
        return headline

    @staticmethod
    def _generate_html(md: str, title: str) -> str:
        # CSS 등은 기존 유지
        css = """<style>body{font-family:sans-serif;line-height:1.6;padding:20px;max-width:800px;margin:auto;color:#333}h1,h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.5em;margin-bottom:1em}h1{font-size:2em}h2{font-size:1.5em}ul,ol{padding-left:2em}li{margin-bottom:.5em}code{background-color:#f0f0f0;padding:.2em .4em;border-radius:3px;font-family:monospace;font-size:.9em}pre{background-color:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}pre code{background-color:transparent;padding:0;border-radius:0}blockquote{border-left:4px solid #ccc;padding-left:1em;color:#666;margin-left:0}table{border-collapse:collapse;width:100%;margin-bottom:1em}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}</style>"""
        try:
            # markdown2 변환은 그대로 사용
            body_html = markdown2.markdown(md, extras=["metadata","fenced-code-blocks","tables","strike","task_list","code-friendly","markdown-in-html"])

            # HTML 제목 결정: 1. Markdown 본문 첫 헤더, 2. 메타데이터 title, 3. 작업명(task)
            html_title = Serializer._extract_headline(md, title) # 본문 헤더 우선
            if hasattr(body_html,"metadata") and body_html.metadata.get("title"):
                 html_title = body_html.metadata["title"] # 메타데이터 있으면 덮어쓰기

            # 제목 단축 (textwrap 임포트 필요)
            safe_title = textwrap.shorten(html_title, width=50, placeholder="...")

            return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{safe_title}</title>{css}</head><body>{body_html}</body></html>"""
        except Exception as e:
            # 오류 처리 로직은 기존 유지 (오류 발생 시 기본 HTML 반환)
            print(f"[yellow]경고: Markdown -> HTML 변환 중 오류 발생: {e}[/]")
            escaped_md = "".join(c if c.isalnum() or c in " .,;:!?/\\#$%&'()*+-=<>[]_{}|`~" else f"&#{ord(c)};" for c in md) # type: ignore
            return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>HTML 생성 오류</title></head><body><h1>HTML 생성 오류</h1><p>Markdown 내용을 표시하는 데 문제가 발생했습니다:</p><pre>{escaped_md}</pre></body></html>"""


    @staticmethod
    def save_state(md: str, task: str, current_app_state_dir: pathlib.Path, current_app_art_dir: pathlib.Path, current_app_root: pathlib.Path) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        # 기존 로직과 거의 동일, headline 추출 방식만 _extract_headline 사용
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        safe_task = "".join(c for c in task if c.isalnum() or c in ('_','-')).strip().replace(' ','_')
        if not safe_task: safe_task="untitled_task"
        base_fn = f"{ts}_{safe_task[:50]}"

        try:
            current_app_state_dir.mkdir(parents=True, exist_ok=True)
            current_app_art_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"상태/아티팩트 디렉토리 생성 실패: {e}") from e

        state_f = current_app_state_dir / f"{base_fn}.md"
        html_f = current_app_state_dir / f"{base_fn}.html"
        meta_f = current_app_state_dir / f"{base_fn}.meta.json"

        try: state_f.write_text(md, encoding="utf-8")
        except IOError as e: raise RuntimeError(f"MD 파일 저장 실패 ({state_f}): {e}") from e

        # HTML 생성 및 저장
        html_content = Serializer._generate_html(md, task) # 작업명을 기본 제목으로 전달
        html_ok = False
        if html_content:
            try:
                 html_f.write_text(html_content, encoding="utf-8")
                 html_ok = True
            except IOError as e: print(f"[yellow]경고: HTML 파일 저장 실패 ({html_f.name}): {e}[/]")
            except Exception as e: print(f"[yellow]경고: HTML 파일 저장 중 예외 ({html_f.name}): {e}[/]")
        else:
             print(f"[yellow]경고: HTML 내용 생성 실패로 '{html_f.name}' 파일 저장을 건너<0xEB><0x9A><0x81>니다.[/yellow]")

        # 아티팩트 스냅샷 로직 유지
        snap_dir = None; checksums = {}
        if current_app_art_dir.exists() and current_app_art_dir.is_dir():
            arts = [f for f in current_app_art_dir.iterdir() if f.is_file()]
            if arts:
                snapshot_sub_dir_name = f"{base_fn}_artifacts"
                snap_dir = current_app_art_dir / snapshot_sub_dir_name
                try: snap_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e: print(f"[yellow]경고: 스냅샷 디렉토리 생성 실패 ({snap_dir}): {e}[/]"); snap_dir=None

                if snap_dir:
                    print(f"[dim]아티팩트 스냅샷 ({len(arts)}개) -> '{snap_dir.relative_to(current_app_root)}'[/]")
                    for f_art in arts:
                        try:
                            target_path = snap_dir / f_art.name
                            shutil.copy2(f_art, target_path)
                            cs = Serializer._calculate_sha256(target_path)
                            if cs: checksums[f_art.name] = cs
                        except Exception as copy_e: print(f"[yellow]경고: 아티팩트 파일 '{f_art.name}' 복사/해시 실패: {copy_e}[/]")

        # 메타데이터 생성 (headline 추출 방식 변경)
        headline = Serializer._extract_headline(md, task) # 작업명을 기본값으로 사용

        meta = {"task":task,"ts":ts,"headline":headline,"artifact_checksums":checksums}
        try: meta_f.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        except IOError as e: raise RuntimeError(f"메타데이터 파일 저장 실패 ({meta_f}): {e}") from e
        except Exception as e: raise RuntimeError(f"메타데이터 JSON 생성/저장 실패: {e}") from e

        # 커밋할 파일 목록 결정
        paths_to_commit = [state_f, meta_f]
        if html_ok and html_f.exists():
             paths_to_commit.append(html_f)

        valid_snap_dir = snap_dir if (snap_dir and snap_dir.exists() and any(snap_dir.iterdir())) else None
        return paths_to_commit, valid_snap_dir

    # to_prompt 메서드 제거 (AI 관련 불필요)
    # @staticmethod
    # def to_prompt(md: str, commit: str) -> str: return f"### 이전 상태 (Commit: {commit}) ###\n\n{md}\n\n### 상태 정보 끝 ###"

# --- UI 클래스 (변경 없음) ---
class UI:
    console = Console()
    @staticmethod
    def task_name(default:str="작업 요약") -> str: return Prompt.ask("[bold cyan]작업 이름[/]",default=default)

    @staticmethod
    def multiline(label: str, default: str = "", help_text: Optional[str] = None) -> str:
        UI.console.print(f"\n[bold cyan]{label}[/]")
        if help_text:
            UI.console.print(f"[dim]{help_text}[/dim]")

        # 수동 모드에서는 default 사용 안 함 (AI 초안 없음)
        # if default:
        #     UI.console.print(Panel(default, title="[dim]제공된 내용 (편집 가능, 완료는 빈 줄에서 Enter 두 번)[/dim]", border_style="dim yellow", expand=False, padding=(0,1)))
        #     UI.console.print("[dim]위 내용을 수정하거나, 그대로 사용하려면 바로 Enter 두 번 입력하세요.[/dim]")
        # else:
        UI.console.print("[dim](입력 완료: 빈 줄에서 Enter 두 번)[/dim]")

        lines = []
        blank_count = 0
        # initial_input_done 변수 불필요
        while True:
            try:
                 line = input("> ") # 항상 프롬프트 표시
            except EOFError: break

            if line == "" :
                blank_count += 1
            else:
                blank_count = 0
                lines.append(line)

            if blank_count >= 2: break

        final_content = "\n".join(lines).strip()

        # 수동 모드에서는 default 사용 로직 불필요
        # if not initial_input_done and default:
        #     print("[dim]입력 없음, 제공된 기본 내용을 사용합니다.[/dim]")
        #     return default
        return final_content # 항상 사용자 입력 반환 (없으면 빈 문자열)

    @staticmethod
    def notify(msg:str,style:str="green"): UI.console.print(f"\n[bold {style}]✔ {msg}[/]")

    @staticmethod
    def error(msg:str,details:Optional[str]=None):
        # 기존 로직 유지
        UI.console.print(f"\n[bold red]❌ 오류: {msg}[/]")
        if details:
            details_lines = details.strip().splitlines()
            max_lines = 15
            display_details = "\n".join(details_lines[:max_lines])
            if len(details_lines) > max_lines: display_details += f"\n[dim]... (총 {len(details_lines)}줄 중 {max_lines}줄 표시)[/dim]"
            UI.console.print(Panel(display_details,title="[dim]상세 정보 (Traceback)[/]",border_style="dim red",expand=False))

    @staticmethod
    def pick_state(states:List[Dict])->Optional[str]:
         # 기존 로직 유지 (최신순 표시)
        if not states: print("[yellow]저장된 상태가 없습니다.[/]"); return None
        tb = Table(title="[bold]저장된 인수인계 상태 목록 (최신순)[/]",box=box.ROUNDED,show_lines=True, expand=False)
        tb.add_column("#",style="dim",justify="right", width=3); tb.add_column("커밋", style="cyan", no_wrap=True, width=10); tb.add_column("작업", style="magenta", min_width=20, overflow="fold"); tb.add_column("시각", style="green", no_wrap=True, width=18); tb.add_column("헤드라인", style="yellow", overflow="fold", min_width=30)
        # states는 이미 최신순 (GitRepo.list_states 결과)
        for i,s in enumerate(states):tb.add_row(str(i),s["hash"],s["task"],s["time"],s["head"])
        UI.console.print(tb)
        choices=[str(i) for i in range(len(states))]
        sel=Prompt.ask("[bold cyan]로드할 상태 번호 (취소하려면 Enter)[/]",choices=choices+[""],default="",show_choices=False)
        if sel.isdigit() and 0 <= int(sel) < len(states):
            selected_state = states[int(sel)]
            print(f"[info]선택된 커밋: {selected_state['hash']} (작업: {selected_state['task']})[/]");
            return selected_state["hash"]
        print("[info]상태 로드를 취소했습니다.[/]"); return None

    @staticmethod
    def panel(txt:str,title:str,border_style:str="blue"): UI.console.print(Panel(txt,title=f"[bold]{title}[/]",border_style=border_style,expand=False,padding=(1,2)))

    @staticmethod
    def diff_panel(txt:str,target:str):
         # 기존 로직 유지
        if not txt or "변경 사항 없음" in txt:
            print(f"[dim]{txt}[/]")
            return
        if txt.startswith("Diff 생성 오류") or txt.startswith("Diff 생성 Git 오류"):
            print(f"[red]{txt}[/]")
            return
        try:
            syntax_obj = Syntax(txt,"diff",theme="dracula",line_numbers=False,word_wrap=False)
            UI.console.print(Panel(syntax_obj,title=f"[bold]Diff (vs {target})[/]",border_style="yellow",expand=True))
        except Exception as e:
            print(f"[red]Diff 출력 중 오류 발생: {e}[/]")
            print(txt)

# --- Handover 클래스 (AI 제거 및 save/load 수정) ---
class Handover:
    def __init__(self, current_app_root: pathlib.Path): # backend_choice 제거
        self.ui = UI()
        self.app_root = current_app_root
        self.state_dir = self.app_root / "ai_states" # 디렉토리 이름은 유지
        self.art_dir = self.app_root / "artifacts"
        self.git: Optional[GitRepo] = None
        # self.ai 제거

        # GitRepo 초기화는 유지
        try:
            git_repo_candidate = GitRepo(self.app_root)
            if git_repo_candidate.repo:
                self.git = git_repo_candidate
        except Exception as e_git_init:
            self.ui.error(f"GitRepo 객체 생성 실패 (Git 기능 사용 불가): {e_git_init}", traceback.format_exc())

        # AIProvider 초기화 로직 제거

    def _ensure_prereqs(self, cmd: str, needs_git: bool): # needs_ai 제거
        # Git 필요 여부만 체크
        if needs_git and (not self.git or not self.git.repo):
            self.ui.error(f"'{cmd}' 명령은 유효한 Git 저장소 내에서 실행해야 합니다."); sys.exit(1)
        # AI 필요 여부 체크 제거

    def save(self):
        self._ensure_prereqs("save", True) # Git만 필요
        try:
            # 작업 이름 결정 로직 유지
            default_task_name = "작업 요약"
            if self.git and self.git.repo:
                current_branch_name = self.git.get_current_branch_name()
                if current_branch_name and not current_branch_name.startswith("DETACHED_HEAD"):
                    default_task_name = current_branch_name
                elif self.git.repo.head.is_valid() and self.git.repo.head.commit:
                     default_task_name = self.git.repo.head.commit.summary
            task_name_input = self.ui.task_name(default=default_task_name)

            # --- AI 관련 로직 제거 ---
            # Git 커밋 정보 수집 및 표시는 사용자 참고용으로 유지 가능 (선택 사항)
            if self.git:
                 self.ui.console.print("\n[dim]참고용: 최근 Git 커밋 정보[/dim]")
                 num_commits_to_fetch = int(os.getenv("HANDOVER_N_COMMITS", 5)) # 표시 개수 줄임
                 recent_commits_data = self.git.collect_recent_commits_info(num_commits=num_commits_to_fetch)
                 if recent_commits_data:
                      # 간단히 로그만 표시
                      log_str = "\n".join(f"- {c['date']} {c['author']}: {c['subject']} ({c['hash'][:7]})"
                                           for c in reversed(recent_commits_data)) # 시간순
                      print(log_str)
                 else:
                      print("[dim]최근 Git 커밋 정보 없음.[/dim]")

            # 사용자 추가 컨텍스트 입력 (AI용 -> 수동 입력용)
            self.ui.console.print("\n[bold green]인수인계 내용을 Markdown 형식으로 작성해주세요.[/bold green]")
            # ui.multiline 호출하여 사용자에게 직접 내용 입력 받기
            final_markdown_content = self.ui.multiline(
                "인수인계 문서 내용 입력", # 라벨 변경
                default="", # 기본값 없음
                help_text="Markdown 문법 사용 가능. 작성이 끝나면 빈 줄에서 Enter 키를 두 번 누르세요."
            )

            if not final_markdown_content.strip():
                self.ui.error("인수인계 문서 내용이 비어있어 저장을 취소합니다."); return

            # --- AI 검증 로직 제거 ---

            # 상태 저장 (Serializer 호출) - 사용자 입력 내용 전달
            saved_state_files, artifact_snapshot_dir = Serializer.save_state(final_markdown_content, task_name_input, self.state_dir, self.art_dir, self.app_root)

            # Git 저장 (GitRepo 호출)
            if not self.git:
                 self.ui.error("Git 저장소가 설정되지 않아 상태를 커밋할 수 없습니다."); return
            commit_short_hash = self.git.save(saved_state_files, task_name_input, artifact_snapshot_dir)
            self.ui.notify(f"인수인계 상태 저장 완료! (Commit: {commit_short_hash})", style="bold green")

            # HTML 프리뷰 경로 안내 (오류 가능성 고려)
            generated_html_file = next((f for f in saved_state_files if f.name.endswith(".html")), None)
            if generated_html_file and generated_html_file.exists():
                try:
                    rel_path = generated_html_file.relative_to(self.app_root)
                    self.ui.console.print(f"[dim]HTML 프리뷰 생성됨: {rel_path}[/dim]")
                except ValueError:
                     self.ui.console.print(f"[dim]HTML 프리뷰 생성됨: {generated_html_file}[/dim]")
            elif generated_html_file and not generated_html_file.exists():
                 self.ui.console.print(f"[yellow]경고: HTML 파일({generated_html_file.name}) 생성이 실패했을 수 있습니다.[/yellow]")


        except Exception as e: self.ui.error(f"Save 작업 중 오류 발생: {str(e)}", traceback.format_exc())


    def load(self, latest: bool = False):
        self._ensure_prereqs("load", True) # Git만 필요
        try:
            if not self.git:
                self.ui.error("Git 저장소가 설정되지 않아 상태를 로드할 수 없습니다."); return

            saved_states_list = self.git.list_states(self.state_dir) # 최신순
            if not saved_states_list: self.ui.error("저장된 인수인계 상태가 없습니다."); return

            selected_commit_hash: Optional[str]
            if latest:
                if not saved_states_list:
                    self.ui.error("저장된 상태가 없어 최근 상태를 로드할 수 없습니다."); return
                selected_commit_hash = saved_states_list[0]["hash"] # 첫 번째가 최신
                print(f"[info]가장 최근 상태 로드 중: {selected_commit_hash} (작업명: {saved_states_list[0]['task']})[/]")
            else: selected_commit_hash = self.ui.pick_state(saved_states_list) # UI 목록은 최신순

            if not selected_commit_hash: return

            self.ui.console.print(f"\n[bold yellow]{selected_commit_hash} 커밋에서 상태 정보를 로드 중입니다...[/]")
            markdown_content = self.git.load_state(selected_commit_hash, self.state_dir)
            self.ui.panel(markdown_content, f"로드된 인수인계 문서 (Commit: {selected_commit_hash})", border_style="cyan")

            # --- AI 보고서 생성 로직 제거 ---

        except Exception as e: self.ui.error(f"Load 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def diff(self, target: str = "HEAD"):
        # 기존 로직 유지 (AI 불필요)
        self._ensure_prereqs("diff", True)
        try:
            if not self.git:
                self.ui.error("Git 저장소가 설정되지 않아 diff를 생성할 수 없습니다."); return
            self.ui.console.print(f"\n[bold yellow]'{target}' 대비 현재 변경 사항을 확인 중입니다... (Git 추적 파일 기준)[/]")
            diff_output_text = self.git.get_diff(target, color=True)
            self.ui.diff_panel(diff_output_text, target)
        except Exception as e: self.ui.error(f"Diff 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def verify_checksums(self, commit_hash: str):
         # 기존 로직 유지 (AI 불필요)
        self._ensure_prereqs("verify", True)
        if not self.git or not self.git.repo :
            self.ui.error("Git 저장소가 설정되지 않아 체크섬을 확인할 수 없습니다."); return

        self.ui.console.print(f"\n[dim]커밋 {commit_hash}의 저장된 아티팩트 체크섬 정보를 표시합니다.[/]")
        try:
            commit_obj = self.git.repo.commit(self.git.repo.git.rev_parse(commit_hash))
            meta_blob: Optional[Blob] = None
            state_dir_rel_path_str = self.git._get_relative_path_str(self.state_dir)
            if not state_dir_rel_path_str:
                self.ui.error(f"상태 디렉토리({self.state_dir})가 Git 저장소 내에 없어 메타데이터를 찾을 수 없습니다."); return

            for item in commit_obj.tree.traverse():
                if isinstance(item, Blob) and \
                   item.path.startswith(state_dir_rel_path_str) and \
                   item.path.endswith(".meta.json"):
                    meta_blob = item; break

            if not meta_blob: self.ui.error(f"커밋 {commit_hash}에서 메타데이터 파일(.meta.json)을 찾을 수 없습니다. (탐색 경로: {state_dir_rel_path_str})"); return

            metadata_content = json.loads(meta_blob.data_stream.read().decode('utf-8'))
            artifact_checksums_data = metadata_content.get("artifact_checksums", {})

            if artifact_checksums_data:
                checksums_pretty_str = json.dumps(artifact_checksums_data, indent=2, ensure_ascii=False)
                self.ui.panel(checksums_pretty_str, f"저장된 아티팩트 체크섬 (Commit: {commit_hash})", border_style="magenta")
            else: print(f"[dim]커밋 {commit_hash}에 저장된 아티팩트 체크섬 정보가 없습니다.[/]")

        except GitCommandError as e: self.ui.error(f"Git 오류: 유효한 커밋 해시가 아니거나 찾을 수 없습니다 ('{commit_hash}'). {e.stderr}")
        except json.JSONDecodeError as e_json: self.ui.error(f"메타데이터 파일 파싱 오류 ({commit_hash}): {e_json}")
        except Exception as e: self.ui.error(f"체크섬 정보 로드/표시 중 오류 ({commit_hash}): {str(e)}", traceback.format_exc())


# --- 스크립트 진입점 ---
def main_cli_entry_point():
    # 경로 설정 및 디렉토리 생성 로직 유지
    cli_root_path = pathlib.Path('.').resolve()
    is_git_repo_at_cli_root = False
    git_repo_root_detected = cli_root_path

    try:
        git_repo_candidate = Repo(str(cli_root_path), search_parent_directories=True)
        git_repo_root_detected = pathlib.Path(git_repo_candidate.working_dir).resolve()
        cli_root_path = git_repo_root_detected
        is_git_repo_at_cli_root = True
    except InvalidGitRepositoryError: pass
    except Exception as e:
        print(f"[yellow]경고: Git 저장소 확인 중 오류 발생 (일부 Git 기능 사용 불가): {e}[/]")

    app_state_dir = cli_root_path / "ai_states" # 디렉토리명 유지
    app_art_dir = cli_root_path / "artifacts"
    try:
        app_state_dir.mkdir(parents=True, exist_ok=True)
        app_art_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[red]오류: 필수 디렉토리 생성 실패 ({app_state_dir} 또는 {app_art_dir}): {e}[/]")
        sys.exit(1)

    # argparse 설정 (AI 백엔드 옵션 제거)
    parser = argparse.ArgumentParser(description="프로젝트 인수인계 상태 관리 도구 (v1.2.0 - 수동 모드)", formatter_class=argparse.RawTextHelpFormatter)

    # backend_choices_list 및 --backend 인자 제거

    subparsers = parser.add_subparsers(dest="command", help="실행할 작업", required=True)

    # 명령어 설정 (AI 필요 여부 제거)
    cmd_configs = [("save", "현재 작업 상태를 수동으로 작성하여 저장", True), # AI 불필요
                   ("load", "과거 저장된 상태 불러오기", True),              # AI 불필요
                   ("diff", "현재 변경 사항 미리보기", True),              # AI 불필요
                   ("verify", "저장된 상태 아티팩트 체크섬 표시", True)]      # AI 불필요

    for name, help_txt, git_req in cmd_configs: # ai_req_flag 제거
        help_suffix = ""
        if git_req: help_suffix += " (Git 필요)"
        # AI 필요 표시 제거
        p = subparsers.add_parser(name, help=f"{help_txt}{help_suffix}")
        if name == "load": p.add_argument("-l", "--latest", action="store_true", help="가장 최근 상태 자동 로드")
        if name == "diff": p.add_argument("target", nargs="?", default="HEAD", help="비교 대상 Commit/Branch (기본값: HEAD)")
        if name == "verify": p.add_argument("commit", help="체크섬 정보를 확인할 상태 커밋 해시")

    args = parser.parse_args()

    # 명령어 유효성 및 Git 필요 여부 검사
    chosen_cmd_config = next((c for c in cmd_configs if c[0] == args.command), None)
    if not chosen_cmd_config:
        UI.error(f"알 수 없는 명령어: {args.command}"); sys.exit(1)
    _, _, git_needed = chosen_cmd_config # ai_needed_for_cmd 제거

    if git_needed and not is_git_repo_at_cli_root:
        UI.error(f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 위치는 Git 저장소가 아님: {pathlib.Path('.').resolve()})"); sys.exit(1)

    # AI 필요 여부 검사 제거

    # 시작 정보 출력 (AI 백엔드 정보 제거)
    print(f"[bold underline]Handover 스크립트 v1.2.0 (수동 모드)[/]")
    if is_git_repo_at_cli_root: print(f"[dim]프로젝트 루트 (Git): {cli_root_path}[/dim]")
    else: print(f"[dim]현재 작업 폴더 (Git 저장소 아님): {cli_root_path}[/dim]")

    # Handover 객체 생성 및 명령어 실행
    try:
        # AI 백엔드 정보 없이 Handover 객체 생성
        handler = Handover(current_app_root=cli_root_path)

        # AI 백엔드 유효성 검사 제거

        # 명령어 실행
        if args.command == "save": handler.save()
        elif args.command == "load": handler.load(latest=args.latest)
        elif args.command == "diff": handler.diff(target=args.target)
        elif args.command == "verify": handler.verify_checksums(commit_hash=args.commit)
    except Exception as e_handler:
        UI.error(f"핸들러 실행 중 예기치 않은 오류: {str(e_handler)}", traceback.format_exc()); sys.exit(1)


if __name__ == "__main__":
    if sys.version_info < (3, 8): print("[bold red]오류: Python 3.8 이상 필요.[/]"); sys.exit(1)
    main_cli_entry_point()