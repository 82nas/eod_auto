#!/usr/bin/env python3
# handover.py – 인수인계 v1.2.2 (수동 작성 모드, 로드 및 타임존 로직 수정)

from __future__ import annotations
import os
import sys
import datetime
import json
import textwrap # HTML 제목 단축 위해 필요
import pathlib
import shutil
import argparse
import hashlib
import subprocess
import re
import traceback
import math # 시간 차이 계산 위해 필요
from typing import List, Dict, Tuple, Optional, Type, Any
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

COMMIT_TAG = "state("

# --- 유틸리티 함수 ---
def parse_ts_str_to_datetime(ts_str: str) -> Optional[datetime.datetime]:
    """YYYYmmddTHHMMSS 형식 문자열을 datetime 객체로 변환"""
    try:
        return datetime.datetime.strptime(ts_str, "%Y%m%dT%H%M%S")
    except (ValueError, TypeError):
        return None

# --- Serializer 클래스 (작업명 정리 메서드 추가) ---
class Serializer:
    @staticmethod
    def _sanitize_task_name(task: str) -> str:
        """파일 이름에 안전하게 사용할 수 있도록 작업 이름을 정리"""
        safe_task = "".join(c for c in task if c.isalnum() or c in ('_','-')).strip().replace(' ','_')
        if not safe_task: safe_task="untitled_task"
        return safe_task[:50] # 길이 제한 추가 (저장 시와 동일하게)

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
        # 변경 없음
        headline = default_title
        for line in md.splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith("# "):
                headline = stripped_line.lstrip('# ').strip()
                break
            elif stripped_line.startswith("## "):
                headline = stripped_line.lstrip('## ').strip()
                break
        return headline

    @staticmethod
    def _generate_html(md: str, title: str) -> str:
        # 변경 없음
        css = """<style>body{font-family:sans-serif;line-height:1.6;padding:20px;max-width:800px;margin:auto;color:#333}h1,h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.5em;margin-bottom:1em}h1{font-size:2em}h2{font-size:1.5em}ul,ol{padding-left:2em}li{margin-bottom:.5em}code{background-color:#f0f0f0;padding:.2em .4em;border-radius:3px;font-family:monospace;font-size:.9em}pre{background-color:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}pre code{background-color:transparent;padding:0;border-radius:0}blockquote{border-left:4px solid #ccc;padding-left:1em;color:#666;margin-left:0}table{border-collapse:collapse;width:100%;margin-bottom:1em}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}</style>"""
        try:
            body_html = markdown2.markdown(md, extras=["metadata","fenced-code-blocks","tables","strike","task_list","code-friendly","markdown-in-html"])
            html_title = Serializer._extract_headline(md, title)
            if hasattr(body_html,"metadata") and body_html.metadata.get("title"):
                html_title = body_html.metadata["title"]
            safe_title = textwrap.shorten(html_title, width=50, placeholder="...")
            return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{safe_title}</title>{css}</head><body>{body_html}</body></html>"""
        except Exception as e:
            print(f"[yellow]경고: Markdown -> HTML 변환 중 오류 발생: {e}[/]")
            escaped_md = "".join(c if c.isalnum() or c in " .,;:!?/\\#$%&'()*+-=<>[]_{}|`~" else f"&#{ord(c)};" for c in md) # type: ignore
            return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>HTML 생성 오류</title></head><body><h1>HTML 생성 오류</h1><p>Markdown 내용을 표시하는 데 문제가 발생했습니다:</p><pre>{escaped_md}</pre></body></html>"""

    @staticmethod
    def save_state(md: str, task: str, current_app_state_dir: pathlib.Path, current_app_art_dir: pathlib.Path, current_app_root: pathlib.Path) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        # UTC 시간 기준으로 타임스탬프 생성하도록 수정
        # ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") # 기존 로컬 시간 기준
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") # *** UTC 시간 기준 ***
        safe_task = Serializer._sanitize_task_name(task) # 분리된 메서드 사용
        base_fn = f"{ts}_{safe_task}" # 길이 제한은 sanitize에서 처리

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

        html_content = Serializer._generate_html(md, task)
        html_ok = False
        if html_content:
            try:
                html_f.write_text(html_content, encoding="utf-8")
                html_ok = True
            except IOError as e: print(f"[yellow]경고: HTML 파일 저장 실패 ({html_f.name}): {e}[/]")
            except Exception as e: print(f"[yellow]경고: HTML 파일 저장 중 예외 ({html_f.name}): {e}[/]")
        else:
            print(f"[yellow]경고: HTML 내용 생성 실패로 '{html_f.name}' 파일 저장을 건너<0xEB><0x9A><0x81>니다.[/yellow]")

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

        headline = Serializer._extract_headline(md, task)
        # 메타데이터에는 task 이름과 UTC 타임스탬프 문자열 저장
        meta = {"task":task,"ts":ts,"headline":headline,"artifact_checksums":checksums}
        try: meta_f.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        except IOError as e: raise RuntimeError(f"메타데이터 파일 저장 실패 ({meta_f}): {e}") from e
        except Exception as e: raise RuntimeError(f"메타데이터 JSON 생성/저장 실패: {e}") from e

        paths_to_commit = [state_f, meta_f]
        if html_ok and html_f.exists():
            paths_to_commit.append(html_f)

        valid_snap_dir = snap_dir if (snap_dir and snap_dir.exists() and any(snap_dir.iterdir())) else None
        return paths_to_commit, valid_snap_dir


# --- GitRepo 클래스 (로드 로직 수정됨 - 이전 버전과 동일하게 유지) ---
class GitRepo:
    def __init__(self, repo_path: pathlib.Path):
        # 변경 없음
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
         # 변경 없음
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
        # 변경 없음
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
        return commits_data

    def _get_diff_summary_for_commit(self, commit_hash: str) -> Dict[str, Any]:
        # 변경 없음
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
        # 변경 없음
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
        # 변경 없음
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

        try: self.repo.index.add(paths_to_add_abs)
        except Exception as e_add: raise RuntimeError(f"Git add 작업 실패: {e_add}")

        commit_msg = f"{COMMIT_TAG}{task})"
        commit_hash = ""
        try:
            commit_obj = self.repo.index.commit(commit_msg)
            commit_hash = commit_obj.hexsha # 전체 해시 저장 후 나중에 단축
        except HookExecutionError as e_hook:
            stderr_msg = str(e_hook.stderr) if hasattr(e_hook, 'stderr') else str(e_hook)
            wsl_error_match = re.search(r"execvpe\(/bin/bash\) failed: No such file or directory", stderr_msg)
            detailed_error = f"Git pre-commit 훅 실패 (WSL bash 실행 오류): {stderr_msg}" if wsl_error_match else f"Git pre-commit 훅 실패 (종료 코드: {e_hook.status}): {stderr_msg}"
            raise RuntimeError(detailed_error) from e_hook
        except GitCommandError as e_commit:
            is_nothing_to_commit = False
            if hasattr(e_commit, 'stderr') and isinstance(getattr(e_commit, 'stderr'), str):
                if "nothing to commit" in getattr(e_commit, 'stderr').lower() or \
                   "no changes added to commit" in getattr(e_commit, 'stderr').lower():
                    is_nothing_to_commit = True
            if is_nothing_to_commit:
                print("[yellow]경고: 커밋할 변경 사항이 없습니다. 이전 상태와 동일합니다.[/yellow]")
                commit_hash = self.repo.head.commit.hexsha + " (변경 없음)" # 이전 커밋 해시 사용
            else: raise RuntimeError(f"Git 커밋 중 오류 발생: {e_commit}") from e_commit
        except Exception as e_commit_other:
            raise RuntimeError(f"Git 커밋 중 예기치 않은 오류 발생: {e_commit_other}") from e_commit_other

        # Push 로직 (변경 없음)
        if self.repo.remotes:
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

        return commit_hash[:8] if commit_hash and " (변경 없음)" not in commit_hash else commit_hash

    def _get_relative_path_str(self, target_path: pathlib.Path) -> Optional[str]:
         # 변경 없음
        if not self.repo_root_path: return None
        try:
            return target_path.resolve().relative_to(self.repo_root_path).as_posix()
        except ValueError: return None
        except Exception: return None

    def _find_best_matching_meta(self, commit: Commit, search_rel_path_str: str) -> Optional[Tuple[Blob, Dict]]:
        """주어진 커밋 내에서 커밋 시간과 가장 가까운 .meta.json 파일을 찾아 반환"""
        if not self.repo: return None

        # 커밋 시간을 UTC로 가져옴 (timezone aware)
        commit_dt_utc = datetime.datetime.fromtimestamp(commit.committed_date, tz=datetime.timezone.utc)
        best_meta_blob: Optional[Blob] = None
        best_meta_data: Optional[Dict] = None
        min_time_diff = float('inf')

        try:
            for item in commit.tree.traverse():
                # 경로와 확장자 확인
                if isinstance(item, Blob) and \
                   item.path.startswith(search_rel_path_str) and \
                   item.path.endswith(".meta.json"):
                    try:
                        # 메타데이터 파싱
                        metadata = json.loads(item.data_stream.read().decode('utf-8'))
                        meta_ts_str = metadata.get("ts")
                        if not meta_ts_str: continue # 타임스탬프 없으면 건너뜀

                        # 메타데이터 타임스탬프 파싱 (naive datetime)
                        meta_dt = parse_ts_str_to_datetime(meta_ts_str)
                        if not meta_dt:
                            # print(f"[yellow]경고: 메타데이터 파일 '{item.path}'의 타임스탬프 형식 오류 무시됨 ('{meta_ts_str}').[/]")
                            continue

                        # 메타데이터 시간을 UTC로 간주하여 timezone aware 객체 생성
                        # (Serializer.save_state에서 UTC로 저장했으므로 이 가정이 맞음)
                        meta_dt_utc = meta_dt.replace(tzinfo=datetime.timezone.utc)

                        # 시간 차이 계산 (UTC 기준)
                        time_diff = abs((commit_dt_utc - meta_dt_utc).total_seconds())

                        # 가장 시간 차이가 적은 메타데이터 선택
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_meta_blob = item
                            best_meta_data = metadata

                    except json.JSONDecodeError:
                        # print(f"[yellow]경고: 메타데이터 파일 '{item.path}' 파싱 오류 무시됨.[/]")
                        continue
                    except Exception as e_inner:
                        # print(f"[yellow]경고: 메타데이터 처리 중 예상치 못한 오류 ({item.path}): {e_inner}[/]")
                        continue
        except Exception as e_traverse:
            print(f"[yellow]경고: 커밋 {commit.hexsha[:7]} 트리 탐색 중 오류: {e_traverse}[/]")
            return None # 탐색 중 오류 발생 시 포기

        # 시간 차이 경고 (필요시 임계값 조정 가능)
        # 이전 버전에서는 이 경고가 타임존 오류 때문에 많이 발생했을 수 있음
        # 타임존 수정 후에는 정상적인 경우 이 경고가 거의 발생하지 않아야 함
        if best_meta_blob and min_time_diff > 60: # 임계값을 60초로 줄여봄
             print(f"[yellow dim]참고: 커밋 {commit.hexsha[:7]} 시간과 메타데이터 시간 차이: {min_time_diff:.0f}초[/]")
             # 필요하다면 여기서 None을 반환하여 아예 못 찾은 것으로 처리할 수도 있음
             # return None
             pass # 일단은 찾은 것으로 간주

        if best_meta_blob and best_meta_data:
            return best_meta_blob, best_meta_data
        else:
            return None

    def list_states(self, current_app_state_dir: pathlib.Path) -> List[Dict]:
        """저장된 상태 목록 반환 (가장 정확한 메타데이터 기반 헤드라인 사용)"""
        if not self.repo: return []
        items = []
        search_rel_path_str = self._get_relative_path_str(current_app_state_dir)
        if not search_rel_path_str:
            print(f"[yellow]경고: 상태 디렉토리({current_app_state_dir})가 Git 저장소 내에 없어 상태 목록을 검색할 수 없습니다.[/]")
            return []

        try:
            # 상태 커밋만 필터링 (메시지 시작 기준)
            state_commits = [c for c in self.repo.iter_commits(max_count=200, first_parent=True)
                             if c.message.startswith(COMMIT_TAG)]
        except Exception as e:
            print(f"[yellow]경고: 상태 커밋 검색 실패: {e}[/]")
            return []

        for c in state_commits:
            headline = "[메타데이터 없음]" # 기본값
            task_name = c.message[len(COMMIT_TAG):-1].strip() # 커밋 메시지에서 task 추출

            # 해당 커밋에서 가장 적합한 메타데이터 찾기
            meta_result = self._find_best_matching_meta(c, search_rel_path_str)

            if meta_result:
                _, metadata = meta_result
                # 메타데이터에서 헤드라인 추출 (없으면 task_name 사용 시도)
                headline = metadata.get("headline") or metadata.get("task", "[헤드라인/작업명 없음]")
            else:
                 # 메타데이터 못찾으면 경고성 헤드라인 사용 가능
                 headline = "[관련 메타파일 없음]"

            items.append({
                "hash": c.hexsha[:8],
                "task": task_name, # 커밋 메시지의 task 사용
                "time": datetime.datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d %H:%M"), # 표시 시간은 로컬 기준
                "head": headline
            })
        return items # 최신 커밋이 앞에 오도록 반환 (iter_commits 결과 순서)

    def load_state(self, commit_hash: str, current_app_state_dir: pathlib.Path) -> str:
        """주어진 커밋 해시에 해당하는 정확한 상태(.md) 파일 내용을 로드"""
        if not self.repo: raise RuntimeError("Git 저장소가 없어 로드할 수 없습니다.")
        try:
            # 전체 해시로 커밋 객체 가져오기 (필요시 rev_parse 사용)
            commit_obj = self.repo.commit(self.repo.git.rev_parse(commit_hash))
        except Exception as e:
            raise RuntimeError(f"커밋 '{commit_hash}' 접근 오류: {e}") from e

        search_rel_path_str = self._get_relative_path_str(current_app_state_dir)
        if not search_rel_path_str:
            raise RuntimeError(f"상태 디렉토리({current_app_state_dir})가 Git 저장소 내에 없습니다.")

        # 1. 해당 커밋에서 가장 적합한 메타데이터 찾기
        meta_result = self._find_best_matching_meta(commit_obj, search_rel_path_str)

        if not meta_result:
            raise RuntimeError(f"커밋 '{commit_hash}'에서 관련 메타데이터 파일(.meta.json)을 찾을 수 없습니다.")

        _, metadata = meta_result
        state_ts_str = metadata.get("ts")
        # state_task = metadata.get("task") # 메타데이터의 task보다 커밋 메시지의 task가 더 신뢰도 높을 수 있음
                                           # 파일명 생성에는 메타데이터의 task 사용 (원래 생성 시 사용된 이름)
        state_task_for_filename = metadata.get("task")


        if not state_ts_str or not state_task_for_filename:
             raise RuntimeError(f"커밋 '{commit_hash}'의 메타데이터 파일에서 'ts' 또는 'task' 정보를 읽을 수 없습니다.")

        # 2. 메타데이터 정보로 정확한 .md 파일 경로 구성
        safe_task = Serializer._sanitize_task_name(state_task_for_filename)
        expected_md_filename = f"{state_ts_str}_{safe_task}.md"
        expected_md_path_str = (pathlib.Path(search_rel_path_str) / expected_md_filename).as_posix()

        # 3. 해당 경로의 .md 파일 찾기
        try:
            # GitPython의 tree 객체에서 직접 경로로 접근 시도
            md_blob = commit_obj.tree / expected_md_path_str
            if isinstance(md_blob, Blob):
                return md_blob.data_stream.read().decode('utf-8')
            else: # 위 방식이 실패하면 traverse로 다시 시도 (안전장치)
                 raise KeyError # KeyError를 발생시켜 아래 except 블록으로 이동

        except KeyError: # tree['path'] 방식 실패 시 또는 위에서 강제 발생시킨 경우
             # traverse 방식으로 다시 시도
             for item in commit_obj.tree.traverse():
                  if isinstance(item, Blob) and item.path == expected_md_path_str:
                       return item.data_stream.read().decode('utf-8')
             # 최종적으로 못 찾은 경우
             raise RuntimeError(f"커밋 '{commit_hash}' 내에서 예상 경로 '{expected_md_path_str}'의 상태 파일(.md)을 찾을 수 없습니다.")
        except Exception as e:
            raise RuntimeError(f"커밋 '{commit_hash}' 상태 로드 중 예기치 않은 오류 ({expected_md_path_str}): {e}") from e


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
        UI.console.print("[dim](입력 완료: 빈 줄에서 Enter 두 번)[/dim]")
        lines = []
        blank_count = 0
        while True:
            try: line = input("> ")
            except EOFError: break
            if line == "" : blank_count += 1
            else: blank_count = 0; lines.append(line)
            if blank_count >= 2: break
        final_content = "\n".join(lines).strip()
        return final_content

    @staticmethod
    def notify(msg:str,style:str="green"): UI.console.print(f"\n[bold {style}]✔ {msg}[/]")

    @staticmethod
    def error(msg:str,details:Optional[str]=None):
        UI.console.print(f"\n[bold red]❌ 오류: {msg}[/]")
        if details:
            details_lines = details.strip().splitlines()
            max_lines = 15
            display_details = "\n".join(details_lines[:max_lines])
            if len(details_lines) > max_lines: display_details += f"\n[dim]... (총 {len(details_lines)}줄 중 {max_lines}줄 표시)[/dim]"
            UI.console.print(Panel(display_details,title="[dim]상세 정보 (Traceback)[/]",border_style="dim red",expand=False))

    @staticmethod
    def pick_state(states:List[Dict])->Optional[str]:
        if not states: print("[yellow]저장된 상태가 없습니다.[/]"); return None
        tb = Table(title="[bold]저장된 인수인계 상태 목록 (최신순)[/]",box=box.ROUNDED,show_lines=True, expand=False)
        tb.add_column("#",style="dim",justify="right", width=3); tb.add_column("커밋", style="cyan", no_wrap=True, width=10); tb.add_column("작업", style="magenta", min_width=20, overflow="fold"); tb.add_column("시각", style="green", no_wrap=True, width=18); tb.add_column("헤드라인", style="yellow", overflow="fold", min_width=30)
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

# --- Handover 클래스 (변경 없음) ---
class Handover:
    def __init__(self, current_app_root: pathlib.Path):
        self.ui = UI()
        self.app_root = current_app_root
        self.state_dir = self.app_root / "ai_states"
        self.art_dir = self.app_root / "artifacts"
        self.git: Optional[GitRepo] = None
        try:
            git_repo_candidate = GitRepo(self.app_root)
            if git_repo_candidate.repo:
                self.git = git_repo_candidate
        except Exception as e_git_init:
            self.ui.error(f"GitRepo 객체 생성 실패 (Git 기능 사용 불가): {e_git_init}", traceback.format_exc())

    def _ensure_prereqs(self, cmd: str, needs_git: bool):
        if needs_git and (not self.git or not self.git.repo):
            self.ui.error(f"'{cmd}' 명령은 유효한 Git 저장소 내에서 실행해야 합니다."); sys.exit(1)

    def save(self):
        self._ensure_prereqs("save", True)
        try:
            default_task_name = "작업 요약"
            if self.git and self.git.repo:
                current_branch_name = self.git.get_current_branch_name()
                if current_branch_name and not current_branch_name.startswith("DETACHED_HEAD"):
                    default_task_name = current_branch_name
                elif self.git.repo.head.is_valid() and self.git.repo.head.commit:
                    # 마지막 커밋 메시지에서 'state(...)' 제외하고 기본값으로 사용 시도
                    last_commit_msg = self.git.repo.head.commit.summary
                    if not last_commit_msg.startswith(COMMIT_TAG):
                        default_task_name = last_commit_msg
                    else: # state 커밋이면 브랜치 이름이나 기본값 사용
                        pass # default_task_name 유지

            task_name_input = self.ui.task_name(default=default_task_name)

            if self.git:
                self.ui.console.print("\n[dim]참고용: 최근 Git 커밋 정보[/dim]")
                num_commits_to_fetch = int(os.getenv("HANDOVER_N_COMMITS", 5))
                recent_commits_data = self.git.collect_recent_commits_info(num_commits=num_commits_to_fetch)
                if recent_commits_data:
                    log_str = "\n".join(f"- {c['date']} {c['author']}: {c['subject']} ({c['hash'][:7]})"
                                        for c in reversed(recent_commits_data))
                    print(log_str)
                else: print("[dim]최근 Git 커밋 정보 없음.[/dim]")

            self.ui.console.print("\n[bold green]인수인계 내용을 Markdown 형식으로 작성해주세요.[/bold green]")
            final_markdown_content = self.ui.multiline(
                "인수인계 문서 내용 입력",
                default="",
                help_text="Markdown 문법 사용 가능. 작성이 끝나면 빈 줄에서 Enter 키를 두 번 누르세요."
            )

            if not final_markdown_content.strip():
                self.ui.error("인수인계 문서 내용이 비어있어 저장을 취소합니다."); return

            saved_state_files, artifact_snapshot_dir = Serializer.save_state(final_markdown_content, task_name_input, self.state_dir, self.art_dir, self.app_root)

            if not self.git:
                self.ui.error("Git 저장소가 설정되지 않아 상태를 커밋할 수 없습니다."); return
            commit_short_hash = self.git.save(saved_state_files, task_name_input, artifact_snapshot_dir)
            self.ui.notify(f"인수인계 상태 저장 완료! (Commit: {commit_short_hash})", style="bold green")

            generated_html_file = next((f for f in saved_state_files if f.name.endswith(".html")), None)
            if generated_html_file and generated_html_file.exists():
                try: rel_path = generated_html_file.relative_to(self.app_root)
                except ValueError: rel_path = generated_html_file # Git root 밖이면 절대 경로 표시
                self.ui.console.print(f"[dim]HTML 프리뷰 생성됨: {rel_path}[/dim]")
            elif generated_html_file and not generated_html_file.exists():
                 self.ui.console.print(f"[yellow]경고: HTML 파일({generated_html_file.name}) 생성이 실패했을 수 있습니다.[/yellow]")

        except Exception as e: self.ui.error(f"Save 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def load(self, latest: bool = False):
        self._ensure_prereqs("load", True)
        try:
            if not self.git:
                self.ui.error("Git 저장소가 설정되지 않아 상태를 로드할 수 없습니다."); return

            saved_states_list = self.git.list_states(self.state_dir)
            if not saved_states_list: self.ui.error("저장된 인수인계 상태가 없습니다."); return

            selected_commit_hash: Optional[str]
            if latest:
                if not saved_states_list:
                    self.ui.error("저장된 상태가 없어 최근 상태를 로드할 수 없습니다."); return
                selected_commit_hash = saved_states_list[0]["hash"]
                print(f"[info]가장 최근 상태 로드 중: {selected_commit_hash} (작업명: {saved_states_list[0]['task']})[/]")
            else: selected_commit_hash = self.ui.pick_state(saved_states_list)

            if not selected_commit_hash: return

            self.ui.console.print(f"\n[bold yellow]{selected_commit_hash} 커밋에서 상태 정보를 로드 중입니다...[/]")
            markdown_content = self.git.load_state(selected_commit_hash, self.state_dir)
            self.ui.panel(markdown_content, f"로드된 인수인계 문서 (Commit: {selected_commit_hash})", border_style="cyan")

        except Exception as e: self.ui.error(f"Load 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def diff(self, target: str = "HEAD"):
        # 변경 없음
        self._ensure_prereqs("diff", True)
        try:
            if not self.git:
                self.ui.error("Git 저장소가 설정되지 않아 diff를 생성할 수 없습니다."); return
            self.ui.console.print(f"\n[bold yellow]'{target}' 대비 현재 변경 사항을 확인 중입니다... (Git 추적 파일 기준)[/]")
            diff_output_text = self.git.get_diff(target, color=True)
            self.ui.diff_panel(diff_output_text, target)
        except Exception as e: self.ui.error(f"Diff 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def verify_checksums(self, commit_hash: str):
         # 변경 없음 (이제 정확한 메타데이터를 찾아서 보여줄 것임)
        self._ensure_prereqs("verify", True)
        if not self.git or not self.git.repo :
            self.ui.error("Git 저장소가 설정되지 않아 체크섬을 확인할 수 없습니다."); return

        self.ui.console.print(f"\n[dim]커밋 {commit_hash}의 저장된 아티팩트 체크섬 정보를 표시합니다.[/]")
        try:
            # 전체 해시로 커밋 객체 가져오기
            commit_obj = self.git.repo.commit(self.git.repo.git.rev_parse(commit_hash))
            state_dir_rel_path_str = self.git._get_relative_path_str(self.state_dir)
            if not state_dir_rel_path_str:
                self.ui.error(f"상태 디렉토리({self.state_dir})가 Git 저장소 내에 없어 메타데이터를 찾을 수 없습니다."); return

            # 정확한 메타데이터 찾기
            meta_result = self.git._find_best_matching_meta(commit_obj, state_dir_rel_path_str)

            if not meta_result:
                self.ui.error(f"커밋 {commit_hash}에서 관련 메타데이터 파일(.meta.json)을 찾을 수 없습니다."); return

            _, metadata_content = meta_result
            artifact_checksums_data = metadata_content.get("artifact_checksums", {})

            if artifact_checksums_data:
                checksums_pretty_str = json.dumps(artifact_checksums_data, indent=2, ensure_ascii=False)
                self.ui.panel(checksums_pretty_str, f"저장된 아티팩트 체크섬 (Commit: {commit_hash})", border_style="magenta")
            else: print(f"[dim]커밋 {commit_hash}에 저장된 아티팩트 체크섬 정보가 없습니다.[/]")

        except GitCommandError as e: self.ui.error(f"Git 오류: 유효한 커밋 해시가 아니거나 찾을 수 없습니다 ('{commit_hash}'). {e.stderr}")
        except json.JSONDecodeError as e_json: self.ui.error(f"메타데이터 파일 파싱 오류 ({commit_hash}): {e_json}") # _find_best_matching_meta에서 처리되지만 안전장치
        except Exception as e: self.ui.error(f"체크섬 정보 로드/표시 중 오류 ({commit_hash}): {str(e)}", traceback.format_exc())


# --- 스크립트 진입점 (버전 업데이트) ---
def main_cli_entry_point():
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

    app_state_dir = cli_root_path / "ai_states"
    app_art_dir = cli_root_path / "artifacts"
    try:
        app_state_dir.mkdir(parents=True, exist_ok=True)
        app_art_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[red]오류: 필수 디렉토리 생성 실패 ({app_state_dir} 또는 {app_art_dir}): {e}[/]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="프로젝트 인수인계 상태 관리 도구 (v1.2.2 - 수동 모드, 로드/타임존 수정)", formatter_class=argparse.RawTextHelpFormatter) # 버전 업데이트
    subparsers = parser.add_subparsers(dest="command", help="실행할 작업", required=True)

    cmd_configs = [("save", "현재 작업 상태를 수동으로 작성하여 저장", True),
                   ("load", "과거 저장된 상태 불러오기", True),
                   ("diff", "현재 변경 사항 미리보기", True),
                   ("verify", "저장된 상태 아티팩트 체크섬 표시", True)]

    for name, help_txt, git_req in cmd_configs:
        help_suffix = " (Git 필요)" if git_req else ""
        p = subparsers.add_parser(name, help=f"{help_txt}{help_suffix}")
        if name == "load": p.add_argument("-l", "--latest", action="store_true", help="가장 최근 상태 자동 로드")
        if name == "diff": p.add_argument("target", nargs="?", default="HEAD", help="비교 대상 Commit/Branch (기본값: HEAD)")
        if name == "verify": p.add_argument("commit", help="체크섬 정보를 확인할 상태 커밋 해시")

    args = parser.parse_args()

    chosen_cmd_config = next((c for c in cmd_configs if c[0] == args.command), None)
    if not chosen_cmd_config:
        UI.error(f"알 수 없는 명령어: {args.command}"); sys.exit(1)
    _, _, git_needed = chosen_cmd_config

    if git_needed and not is_git_repo_at_cli_root:
        UI.error(f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 위치는 Git 저장소가 아님: {pathlib.Path('.').resolve()})"); sys.exit(1)

    print(f"[bold underline]Handover 스크립트 v1.2.2 (수동 모드, 로드/타임존 수정)[/]") # 버전 업데이트
    if is_git_repo_at_cli_root: print(f"[dim]프로젝트 루트 (Git): {cli_root_path}[/dim]")
    else: print(f"[dim]현재 작업 폴더 (Git 저장소 아님): {cli_root_path}[/dim]")

    try:
        handler = Handover(current_app_root=cli_root_path)
        if args.command == "save": handler.save()
        elif args.command == "load": handler.load(latest=args.latest)
        elif args.command == "diff": handler.diff(target=args.target)
        elif args.command == "verify": handler.verify_checksums(commit_hash=args.commit)
    except Exception as e_handler:
        UI.error(f"핸들러 실행 중 예기치 않은 오류: {str(e_handler)}", traceback.format_exc()); sys.exit(1)

if __name__ == "__main__":
    if sys.version_info < (3, 8): print("[bold red]오류: Python 3.8 이상 필요.[/]"); sys.exit(1)
    main_cli_entry_point()