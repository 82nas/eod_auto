#!/usr/bin/env python3
# handover.py – 인수인계 v1.1.6 (백엔드 로딩 및 Git 경로 처리 최종 수정)

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
import importlib.util # 명시적 임포트 추가
from typing import List, Dict, Tuple, Optional, Type, Any
from dotenv import load_dotenv

# --- 의존성 로드 ---
try:
    from git import Repo, GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Blob, Commit
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
    print("팁: [yellow]pip install gitpython requests rich python-dotenv markdown2[/] 명령을 실행하세요.")
    sys.exit(1)

load_dotenv()

# backends 폴더는 handover.py 파일과 같은 디렉토리에 있다고 가정
BACKEND_DIR = pathlib.Path(__file__).resolve().parent / "backends"
COMMIT_TAG = "state("

# --- AI 백엔드 로딩 ---
AIBaseBackend = None # 기본값 설정
try:
    base_spec = importlib.util.spec_from_file_location("backends.base", BACKEND_DIR / "base.py")
    if base_spec is None or base_spec.loader is None:
        raise ImportError(f"backends.base 모듈 스펙을 찾을 수 없습니다. 경로: {BACKEND_DIR / 'base.py'}")
    backends_base_module = importlib.util.module_from_spec(base_spec)
    sys.modules['backends.base'] = backends_base_module # sys.modules에 명시적으로 추가
    base_spec.loader.exec_module(backends_base_module)
    AIBaseBackend = backends_base_module.AIBaseBackend
except ImportError as e:
    print(f"[bold red]오류: backends.base 모듈 임포트 실패: {e}[/]")
    # AIBaseBackend가 None으로 유지되어 아래 available_backends가 비게 됨
except AttributeError:
    print(f"[bold red]오류: backends.base 모듈에서 AIBaseBackend 클래스 찾기 실패.[/]")
except FileNotFoundError:
    print(f"[bold red]오류: backends/base.py 파일을 찾을 수 없습니다. 경로: {BACKEND_DIR / 'base.py'}[/]")

available_backends: Dict[str, Type[AIBaseBackend]] = {}
if AIBaseBackend and BACKEND_DIR.exists() and BACKEND_DIR.is_dir(): # AIBaseBackend가 성공적으로 로드된 경우에만 진행
    for f_py in BACKEND_DIR.glob("*.py"):
        module_name_stem = f_py.stem
        if module_name_stem == "__init__" or module_name_stem == "base":
            continue
        try:
            # 각 백엔드 모듈의 전체 이름 (예: backends.ollama)
            full_module_name = f"backends.{module_name_stem}"
            spec = importlib.util.spec_from_file_location(full_module_name, f_py)
            if spec is None or spec.loader is None:
                print(f"[yellow]경고: 백엔드 모듈 스펙 로딩 실패 {f_py.name}[/]")
                continue
            module = importlib.util.module_from_spec(spec)
            # sys.modules에 추가해야 해당 모듈 내에서 from .base import ... 같은 상대 경로 임포트가 잘 동작함
            sys.modules[full_module_name] = module
            spec.loader.exec_module(module)

            for name, obj in module.__dict__.items():
                if (isinstance(obj, type) and
                        # AIBaseBackend 자체가 None일 수 있으므로 issubclass 전에 체크
                        AIBaseBackend and issubclass(obj, AIBaseBackend) and
                        obj is not AIBaseBackend):
                    backend_name_from_class = obj.get_name()
                    if backend_name_from_class != "base": # "base"는 실제 백엔드 이름이 아님
                        available_backends[backend_name_from_class] = obj
        except ImportError as e: # 모듈 내부 임포트 오류 등
            print(f"[yellow]경고: 백엔드 모듈 '{module_name_stem}' 로딩 중 임포트 오류: {e}[/]")
        except AttributeError as e:
             print(f"[yellow]경고: 백엔드 클래스 '{name if 'name' in locals() else module_name_stem}' 속성 오류: {e}[/]")
        except Exception as e:
            print(f"[yellow]경고: 백엔드 파일 '{f_py.name}' 처리 중 예외 발생: {e}[/]")
elif not AIBaseBackend:
    print(f"[yellow]경고: AIBaseBackend 로딩 실패로 백엔드 목록을 만들 수 없습니다.[/]")
else: # BACKEND_DIR 문제
    print(f"[yellow]경고: 백엔드 디렉토리 '{BACKEND_DIR}'를 찾을 수 없거나 디렉토리가 아닙니다.[/]")


# --- AIProvider 클래스 ---
class AIProvider:
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        if not AIBaseBackend: # Base 클래스 로딩 실패 시 초기화 불가
             raise RuntimeError("AIProvider 초기화 실패: AIBaseBackend 로딩 실패.")

        if not available_backends and backend_name != "none": # "none"이 아니고 사용 가능한 백엔드가 없을 때
             raise RuntimeError("AIProvider 초기화 실패: 사용 가능한 AI 백엔드가 없습니다.")

        if backend_name == "none": # AI 기능 비활성화 선택
            self.backend = None
            # main_cli_entry_point에서 필요한 경우 메시지 출력
            return

        if backend_name not in available_backends: # 선택한 백엔드가 목록에 없을 때
            raise ValueError(f"알 수 없는 백엔드: '{backend_name}'. 사용 가능: {list(available_backends.keys()) + ['none']}")

        BackendClass = available_backends[backend_name]
        try:
            self.backend: Optional[AIBaseBackend] = BackendClass(config) # 타입 힌트 명시
            # 성공 메시지는 main_cli_entry_point에서 출력
        except Exception as e: # 백엔드 클래스 초기화 중 예외 발생 시
            print(f"[bold red]오류: 백엔드 '{backend_name}' 초기화 실패: {e}[/]")
            if hasattr(BackendClass, 'get_config_description'): # 설정 정보가 있다면 출력
                print(f"[yellow]필요 설정:\n{BackendClass.get_config_description()}[/]")
            raise e # 오류를 다시 발생시켜 Handover에서 처리하도록 함

    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        if not self.backend: raise RuntimeError("AI 백엔드가 'none'으로 설정되어 요약을 생성할 수 없습니다.")
        return self.backend.make_summary(task, ctx, arts)

    def verify_summary(self, md: str) -> Tuple[bool, str]:
        if not self.backend: raise RuntimeError("AI 백엔드가 'none'으로 설정되어 요약을 검증할 수 없습니다.")
        is_ok, msg = self.backend.verify_summary(md)
        # AI가 OK라고 해도 기본적인 구조 검증 추가
        if is_ok:
           lines = md.strip().split('\n')
           headers = [l.strip() for l in lines if l.startswith('#')]
           required_headers_structure = ["#", "## 목표", "## 진행", "## 결정", "## 결과", "## 다음할일", "## 산출물"]
           if len(headers) != len(required_headers_structure): # 정확히 7개 헤더인지 확인
               is_ok = False
               msg = f"헤더 개수 불일치 (필수 {len(required_headers_structure)}개, 현재 {len(headers)}개)"
           else:
               for i, expected_start_format in enumerate(required_headers_structure):
                   # 첫 번째 헤더는 '# 작업이름' 형태이므로 '# '로 시작하는지만 확인
                   if i == 0 and not headers[i].startswith("# "):
                       is_ok = False
                       msg = f"첫 번째 헤더 형식 오류: '{headers[i]}' (예상: '# 작업이름')"
                       break
                   # 나머지 헤더는 '## 헤더명' 형태 (뒤에 공백 필수)
                   elif i > 0 and not headers[i].startswith(expected_start_format + " "):
                       is_ok = False
                       msg = f"헤더 #{i+1} 형식 또는 순서 오류: '{headers[i]}' (예상: '{expected_start_format} 이름')"
                       break
        return is_ok, msg

    def load_report(self, md: str) -> str:
        if not self.backend: raise RuntimeError("AI 백엔드가 'none'으로 설정되어 보고서를 로드할 수 없습니다.")
        return self.backend.load_report(md)

# --- GitRepo 클래스 ---
class GitRepo:
    def __init__(self, repo_path: pathlib.Path):
        self.repo: Optional[Repo] = None # 초기화
        try:
            self.repo = Repo(repo_path)
            if self.repo.bare:
                print(f"[yellow]경고: '{repo_path}'는 bare 저장소입니다. 작업 디렉토리 분석이 제한될 수 있습니다.[/]")
                # Bare 저장소도 커밋 로그 등 일부 작업은 가능하므로 초기화는 진행
        except InvalidGitRepositoryError:
            # Handover 클래스 생성자에서 self.git = None 으로 처리됨 (여기서 raise 불필요)
            pass
        except NoSuchPathError:
             print(f"[red]오류: Git 저장소 경로를 찾을 수 없습니다: {repo_path}[/]")
             # Handover 클래스 생성자에서 self.git = None 으로 처리됨
             pass
        except GitCommandError as e:
            print(f"[red]오류: Git 명령어 실행 실패 ({e.command}). Git이 설치되어 있고 PATH에 있는지 확인하세요.[/]")
            print(f"[dim]상세: {e.stderr}[/dim]")
            # Handover 클래스 생성자에서 self.git = None 으로 처리됨
            pass
        except Exception as e: # 기타 예외
            print(f"[red]오류: Git 저장소 초기화 중 예외 발생: {e}[/]")
            # Handover 클래스 생성자에서 self.git = None 으로 처리됨
            pass

    def _safe(self, git_func, *args, **kwargs):
        if not self.repo: raise RuntimeError("GitRepo 객체가 초기화되지 않았습니다.")
        try: return git_func(*args, **kwargs)
        except GitCommandError as e: stderr = e.stderr.strip(); raise RuntimeError(f"Git 명령어 실패: {e.command}\n오류: {stderr}") from e
        except Exception as e: raise RuntimeError(f"Git 작업 중 예외 발생: {e}") from e

    def get_last_state_commit(self) -> Optional[Commit]:
        if not self.repo: return None
        try:
            for c in self.repo.iter_commits(max_count=200, first_parent=True):
                if c.message.startswith(COMMIT_TAG): return c
        except Exception: pass
        return None

    def get_commit_messages_since(self, commit_hash: Optional[str]) -> str:
        if not self.repo: return "Git 저장소 없음."
        if not commit_hash:
            try:
                commits = list(self.repo.iter_commits(max_count=10, no_merges=True))
                log = "\n".join(f"- {c.hexsha[:7]}: {c.summary}" for c in reversed(commits))
                return f"최근 커밋 {len(commits)}개:\n{log}" if log else "최근 커밋 없음"
            except Exception as e: return f"최근 커밋 로그 조회 실패: {e}"
        try:
            self.repo.commit(commit_hash) # 대상 커밋 유효성 검사
            log_cmd = f"{commit_hash}..HEAD"
            commit_log = self.repo.git.log(log_cmd, '--pretty=format:- %h: %s', '--abbrev-commit', '--no-merges')
            return f"'{commit_hash[:8]}' 이후 커밋:\n{commit_log}" if commit_log else f"'{commit_hash[:8]}' 이후 커밋 없음"
        except GitCommandError as e:
            return f"커밋 로그 조회 실패 ({commit_hash}): {e.stderr}"
        except Exception as e:
            return f"커밋 로그 조회 중 오류 ({commit_hash}): {e}"

    def get_current_branch(self) -> Optional[str]:
        if not self.repo: return "Git 저장소 없음"
        try: return self.repo.active_branch.name
        except TypeError: # Detached HEAD 상태
            try: return f"DETACHED_HEAD@{self.repo.head.commit.hexsha[:7]}"
            except Exception: return "DETACHED_HEAD"
        except Exception: return None # 기타 예외

    def get_diff(self, target: str = "HEAD", color: bool = True) -> str:
        if not self.repo: return "Git 저장소 없음."
        try:
            self.repo.commit(target) # Target 유효성 검사
            color_opt = '--color=always' if color else '--color=never'
            # diff 명령어 실행 시 --exit-code 옵션을 추가하면 변경사항 없을 때 0, 있으면 1 반환 (오류 시 >1)
            # GitPython에서는 직접 exit_code를 받기 어려우므로 결과 문자열로 판단
            staged_diff = self.repo.git.diff('--staged', target, color_opt)
            working_tree_vs_target_diff = self.repo.git.diff(target, color_opt)

            diff_output = ""
            has_staged = bool(staged_diff.strip())
            has_wt_vs_target = bool(working_tree_vs_target_diff.strip())

            if has_staged: diff_output += f"--- Staged Changes (vs {target}) ---\n{staged_diff}\n\n"
            # Working directory 변경 사항은 staged와 다를 경우에만 표시 (중복 방지)
            if has_wt_vs_target and working_tree_vs_target_diff != staged_diff:
                 diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"
            # Staged 변경은 없고 Working directory 변경만 있을 경우 표시
            elif not has_staged and has_wt_vs_target:
                 diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"

            return diff_output.strip() if diff_output.strip() else f"'{target}'과(와) 변경 사항 없음 (Git 추적 파일 기준)"
        except GitCommandError as e: return f"Diff 생성 Git 오류: {e.stderr}"
        except Exception as e: return f"Diff 생성 오류: {e}"

    def save(self, state_paths: List[pathlib.Path], task: str, snapshot_dir: Optional[pathlib.Path]) -> str:
        if not self.repo: raise RuntimeError("Git 저장소가 없어 저장할 수 없습니다.")
        # state_paths 와 snapshot_dir 내 파일 경로를 저장소 루트 기준 상대 경로로 변환하여 add 시도
        paths_to_add_rel = []
        repo_root = pathlib.Path(self.repo.working_dir)
        for p in state_paths:
            try: paths_to_add_rel.append(str(p.resolve().relative_to(repo_root)))
            except ValueError: print(f"[yellow]경고: Git 저장소 외부 경로 추가 시도 무시됨: {p}[/]") # 저장소 외부 파일이면 무시
        if snapshot_dir and snapshot_dir.exists() and any(snapshot_dir.iterdir()):
            try: paths_to_add_rel.append(str(snapshot_dir.resolve().relative_to(repo_root)))
            except ValueError: print(f"[yellow]경고: Git 저장소 외부 스냅샷 경로 추가 시도 무시됨: {snapshot_dir}[/]")

        if not paths_to_add_rel: raise ValueError("저장할 상태 파일 또는 스냅샷 파일이 없습니다 (Git 저장소 내).")

        self._safe(self.repo.git.add, *paths_to_add_rel) # 상대 경로 사용
        commit_msg = f"{COMMIT_TAG}{task})"
        try:
            self._safe(self.repo.index.commit, commit_msg)
            commit_hash = self.repo.head.commit.hexsha[:8]
        except RuntimeError as e:
            err_str = str(e).lower()
            if "nothing to commit" in err_str or \
               "no changes added to commit" in err_str or \
               "changes not staged for commit" in err_str :
                print("[yellow]경고: 커밋할 변경 사항이 없습니다. 이전 상태와 동일할 수 있습니다.[/]")
                commit_hash = self.repo.head.commit.hexsha[:8] + " (변경 없음)"
            else:
                raise e # 다른 종류의 RuntimeError는 다시 발생시킴

        # Push 로직 (정상 커밋 또는 변경 없음 시 모두 시도 가능)
        if self.repo.remotes:
            try:
                current_branch_name = self.get_current_branch()
                if current_branch_name and not current_branch_name.startswith("DETACHED_HEAD"):
                    print(f"[dim]'{current_branch_name}' 브랜치를 원격 저장소(origin)에 푸시 시도...[/]")
                    self._safe(self.repo.git.push, 'origin', current_branch_name)
                    print("[green]원격 저장소에 푸시 완료.[/]")
                else:
                    print(f"[yellow]경고: 현재 브랜치({current_branch_name})가 특정되지 않았거나 Detached HEAD 상태이므로 푸시를 건너뜁니다.[/]")
            except RuntimeError as e:
                print(f"[yellow]경고: 원격 저장소 푸시 실패. 로컬에는 커밋되었습니다. ({e})[/]")
            except Exception as e_general: # 다른 예외 (네트워크 오류 등)
                print(f"[yellow]경고: 원격 저장소 푸시 중 예기치 않은 오류: {e_general}[/]")
        else:
            print("[yellow]경고: 설정된 원격 저장소가 없어 푸시를 건너뜁니다.[/]")

        return commit_hash # 최종 커밋 해시 반환 (변경 없음 포함)

    def _get_relative_path_str(self, target_path: pathlib.Path) -> Optional[str]:
        """주어진 절대 경로를 Git 저장소 루트 기준 상대 경로 문자열로 변환"""
        if not self.repo: return None
        try:
            # Posix 스타일 경로로 변환 (Git 내부 경로와 일치시키기 위함)
            return target_path.resolve().relative_to(self.repo.working_dir).as_posix()
        except ValueError:
            # target_path가 repo 외부에 있을 경우
            return None
        except Exception: # 혹시 모를 다른 경로 관련 에러
            return None

    def list_states(self, current_app_state_dir: pathlib.Path) -> List[Dict]:
        if not self.repo: return []
        items = []
        # 검색할 경로를 저장소 루트 기준 상대 경로로 변환
        search_rel_path_str = self._get_relative_path_str(current_app_state_dir)
        if not search_rel_path_str:
             print(f"[yellow]경고: 상태 디렉토리({current_app_state_dir})가 Git 저장소 내에 없어 상태 목록을 검색할 수 없습니다.[/]")
             return []

        try:
            # iter_commits의 paths 인자에는 저장소 루트 기준 상대 경로 전달
            commits = list(self.repo.iter_commits(max_count=100, first_parent=True, paths=search_rel_path_str))
        except Exception as e:
            print(f"[yellow]경고: 특정 경로({search_rel_path_str}) 커밋 검색 실패, 전체 커밋에서 검색 시도: {e}[/]")
            commits = list(self.repo.iter_commits(max_count=100, first_parent=True))

        for c in commits:
            if not c.message.startswith(COMMIT_TAG): continue
            headline = ""; meta_blob = None
            try:
                # 커밋 트리 내에서 메타파일(.meta.json) 찾기
                for item in c.tree.traverse():
                    # item.path는 저장소 루트 기준 상대 경로 (posix 스타일)
                    if isinstance(item, Blob) and \
                       item.path.startswith(search_rel_path_str) and \
                       item.path.endswith(".meta.json"):
                        meta_blob = item
                        break # 첫 번째 찾은 메타파일 사용

                if meta_blob:
                    metadata = json.loads(meta_blob.data_stream.read().decode('utf-8'))
                    headline = metadata.get("headline", "")
                # else: # 메타 파일이 없어도 커밋 메시지 기반으로 항목은 추가
                #    headline = "[메타파일 없음]"
            except Exception as e: headline = f"[메타데이터 오류: {e}]"
            items.append({"hash": c.hexsha[:8], "task": c.message[len(COMMIT_TAG):-1].strip(), "time": datetime.datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d %H:%M"), "head": headline or "-"})
        return list(reversed(items))

    def load_state(self, commit_hash: str, current_app_state_dir: pathlib.Path) -> str:
        if not self.repo: raise RuntimeError("Git 저장소가 없어 로드할 수 없습니다.")
        try: commit_obj = self.repo.commit(self.repo.git.rev_parse(commit_hash))
        except Exception as e: raise RuntimeError(f"커밋 '{commit_hash}' 접근 오류: {e}") from e

        # 로드할 상태 파일 경로 (저장소 루트 기준 상대 경로)
        search_rel_path_str = self._get_relative_path_str(current_app_state_dir)
        if not search_rel_path_str:
             raise RuntimeError(f"상태 디렉토리({current_app_state_dir})가 Git 저장소 내에 없습니다.")

        try:
            # 커밋 트리 내에서 상태 파일(.md) 찾기
            for item in commit_obj.tree.traverse():
                if isinstance(item, Blob) and \
                   item.path.startswith(search_rel_path_str) and \
                   item.path.endswith(".md"):
                    return item.data_stream.read().decode('utf-8')

            raise RuntimeError(f"커밋 '{commit_hash}' 내 경로 '{search_rel_path_str}'에서 상태 파일(.md)을 찾을 수 없습니다.")
        except Exception as e:
            raise RuntimeError(f"커밋 '{commit_hash}' 상태 로드 중 예기치 않은 오류: {e}")

# --- Serializer 클래스 ---
class Serializer:
    @staticmethod
    def _calculate_sha256(fp: pathlib.Path) -> Optional[str]: # 올바르게 수정된 버전
        h = hashlib.sha256()
        try:
            with open(fp, "rb") as f:
                while True:
                    b = f.read(4096)
                    if not b:
                        break
                    h.update(b)
            return h.hexdigest()
        except IOError:
            # print(f"[yellow]경고: 파일 해시 계산 중 IO오류 ({fp.name})[/]") # 필요시 주석 해제
            return None
        except Exception as e:
            # print(f"[yellow]경고: 파일 해시 계산 중 예외 ({fp.name}): {e}[/]") # 필요시 주석 해제
            return None

    @staticmethod
    def _generate_html(md: str, title: str) -> str:
        css = """<style>body{font-family:sans-serif;line-height:1.6;padding:20px;max-width:800px;margin:auto;color:#333}h1,h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.5em;margin-bottom:1em}h1{font-size:2em}h2{font-size:1.5em}ul,ol{padding-left:2em}li{margin-bottom:.5em}code{background-color:#f0f0f0;padding:.2em .4em;border-radius:3px;font-family:monospace;font-size:.9em}pre{background-color:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}pre code{background-color:transparent;padding:0;border-radius:0}blockquote{border-left:4px solid #ccc;padding-left:1em;color:#666;margin-left:0}table{border-collapse:collapse;width:100%;margin-bottom:1em}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}</style>"""
        try:
             body = markdown2.markdown(md, extras=["metadata","fenced-code-blocks","tables","strike","task_list","code-friendly","markdown-in-html"])
             title_meta = title
             if hasattr(body,"metadata") and body.metadata.get("title"): title_meta = body.metadata["title"]
             return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{textwrap.shorten(title_meta, width=50, placeholder="...")}</title>{css}</head><body>{body}</body></html>"""
        except Exception as e:
             print(f"[yellow]경고: Markdown -> HTML 변환 중 오류 발생: {e}[/]")
             # 오류 발생 시 간단한 HTML 반환
             escaped_md = "".join(c if c.isalnum() or c in " .,;:!?/\\#$%&'()*+-=<>[]_{}|`~" else f"&#{ord(c)};" for c in md) # Basic escaping
             return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>HTML 생성 오류</title></head><body><h1>HTML 생성 오류</h1><p>Markdown 내용을 표시하는 데 문제가 발생했습니다:</p><pre>{escaped_md}</pre></body></html>"""


    @staticmethod
    def save_state(md: str, task: str, current_app_state_dir: pathlib.Path, current_app_art_dir: pathlib.Path, current_app_root: pathlib.Path) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S"); safe_task = "".join(c for c in task if c.isalnum() or c in (' ','_','-')).strip().replace(' ','_');
        if not safe_task: safe_task="untitled_task";
        base_fn = f"{ts}_{safe_task}"
        # 디렉토리는 항상 생성 시도 (오류 시 아래 파일 쓰기에서 문제 발생)
        current_app_state_dir.mkdir(parents=True, exist_ok=True)
        current_app_art_dir.mkdir(parents=True, exist_ok=True)

        state_f = current_app_state_dir / f"{base_fn}.md"
        html_f = current_app_state_dir / f"{base_fn}.html"
        meta_f = current_app_state_dir / f"{base_fn}.meta.json"

        try: state_f.write_text(md, encoding="utf-8")
        except IOError as e: raise RuntimeError(f"MD 파일 저장 실패 ({state_f}): {e}") from e

        html_content = Serializer._generate_html(md, task)
        html_ok = False
        if html_content: # HTML 생성 성공 시 저장
            try:
                html_f.write_text(html_content, encoding="utf-8")
                html_ok = True
            except IOError as e: print(f"[yellow]경고: HTML 파일 저장 실패 ({html_f.name}): {e}[/]")
            except Exception as e: print(f"[yellow]경고: HTML 파일 저장 중 예외 ({html_f.name}): {e}[/]")

        snap_dir = None; checksums = {}
        # artifacts 폴더가 존재하고, 그 안에 파일이 있을 경우에만 스냅샷 시도
        if current_app_art_dir.exists() and current_app_art_dir.is_dir():
            arts = [f for f in current_app_art_dir.iterdir() if f.is_file()]
            if arts:
                snapshot_sub_dir_name = f"{base_fn}_artifacts"
                snap_dir = current_app_art_dir / snapshot_sub_dir_name
                snap_dir.mkdir(parents=True, exist_ok=True) # 스냅샷 하위 폴더 생성
                print(f"[dim]아티팩트 스냅샷 ({len(arts)}개) -> '{snap_dir.relative_to(current_app_root)}'[/]")
                for f_art in arts:
                    try:
                        target_path = snap_dir / f_art.name
                        shutil.copy2(f_art, target_path) # 메타데이터 포함 복사
                        cs = Serializer._calculate_sha256(target_path)
                        if cs: checksums[f_art.name] = cs
                    except Exception as copy_e: print(f"[yellow]경고: 아티팩트 파일 '{f_art.name}' 복사/해시 실패: {copy_e}[/]")
            else: print("[dim]아티팩트 폴더에 파일이 없어 스냅샷을 건너<0xEB><0x9B><0x81>니다.[/]")
        else: print(f"[dim]아티팩트 폴더({current_app_art_dir})가 없거나 디렉토리가 아닙니다.[/]")

        headline = task # 기본값은 태스크 이름
        for ln in md.splitlines():
            ln_s = ln.strip()
            if ln_s.startswith("#"): # 첫 번째 '#' 라인을 헤드라인으로
                headline = ln_s.lstrip('# ').strip()
                break

        meta = {"task":task,"ts":ts,"headline":headline,"artifact_checksums":checksums}
        try: meta_f.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        except IOError as e: raise RuntimeError(f"메타데이터 파일 저장 실패 ({meta_f}): {e}") from e
        except Exception as e: raise RuntimeError(f"메타데이터 JSON 생성/저장 실패: {e}") from e

        paths_to_commit = [state_f, meta_f]
        if html_ok and html_f.exists(): paths_to_commit.append(html_f)
        # 스냅샷 디렉토리가 생성되었고, 안에 파일이 실제 복사된 경우만 Git에 추가할 디렉토리로 간주
        valid_snap_dir = snap_dir if (snap_dir and snap_dir.exists() and any(snap_dir.iterdir())) else None
        return paths_to_commit, valid_snap_dir

    @staticmethod
    def to_prompt(md: str, commit: str) -> str: return f"### 이전 상태 (Commit: {commit}) ###\n\n{md}\n\n### 상태 정보 끝 ###"

# --- UI 클래스 ---
class UI:
    console = Console()
    @staticmethod
    def task_name(default:str="작업 요약") -> str: return Prompt.ask("[bold cyan]작업 이름[/]",default=default)
    @staticmethod
    def multiline(label: str, default: str = "") -> str:
        UI.console.print(f"\n[bold cyan]{label}[/]"); UI.console.print("[dim](입력 완료: 빈 줄에서 Enter 두 번)[/]")
        lines = []
        if default:
            # 기본값 미리보기를 좀 더 안전하게 처리 (너무 길면 줄임)
            default_preview = textwrap.shorten(default.splitlines()[0], width=100, placeholder="...") if default else ""
            if len(default.splitlines()) > 1: default_preview += " (...)"
            print(Panel(default_preview,title="[dim]자동 제안 (편집 가능, 전체 내용은 아래 입력)[/]",border_style="dim",expand=False))
        blank_count = 0
        while True:
            try: line = input()
            except EOFError: break
            if line=="": blank_count+=1
            else: blank_count=0 # 내용 입력 시 카운트 리셋
            if blank_count>=2: break # 빈 줄 두 번 연속 시 종료
            lines.append(line) # 빈 줄 포함하여 라인 추가
        final = "\n".join(lines).strip()
        if not final and default: print("[dim]입력 없음, 자동 제안 내용을 사용합니다.[/]"); return default
        return final
    @staticmethod
    def notify(msg:str,style:str="green"): UI.console.print(f"\n[bold {style}]✔ {msg}[/]")
    @staticmethod
    def error(msg:str,details:Optional[str]=None):
        UI.console.print(f"\n[bold red]❌ 오류: {msg}[/]")
        if details:
             # Traceback이 너무 길 경우 잘라서 표시
             details_lines = details.strip().splitlines()
             max_lines = 15
             display_details = "\n".join(details_lines[:max_lines])
             if len(details_lines) > max_lines: display_details += "\n[dim]... (이하 생략)[/dim]"
             UI.console.print(Panel(display_details,title="[dim]상세 정보 (Traceback)[/]",border_style="dim red",expand=False))
    @staticmethod
    def pick_state(states:List[Dict])->Optional[str]:
        if not states: print("[yellow]저장된 상태가 없습니다.[/]"); return None
        tb = Table(title="[bold]저장된 인수인계 상태 목록[/]",box=box.ROUNDED,show_lines=True, expand=False)
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
             print(f"[red]{txt}[/]") # 오류는 빨간색으로 표시
             return

        try:
            syntax_obj = Syntax(txt,"diff",theme="default",line_numbers=False,word_wrap=False)
            UI.console.print(Panel(syntax_obj,title=f"[bold]Diff (vs {target})[/]",border_style="yellow",expand=True))
        except Exception as e:
             print(f"[red]Diff 출력 중 오류 발생: {e}[/]")
             print(txt) # 원본 텍스트라도 출력

# --- Handover 클래스 ---
class Handover:
    def __init__(self, backend_choice: str, current_app_root: pathlib.Path):
        self.ui = UI(); self.app_root = current_app_root
        self.state_dir = self.app_root / "ai_states"
        self.art_dir = self.app_root / "artifacts"
        self.git: Optional[GitRepo] = None # 타입 힌트 명시
        self.ai: Optional[AIProvider] = None

        try:
            git_repo_candidate = GitRepo(self.app_root)
            # GitRepo 생성자에서 repo 초기화 실패 시 self.repo가 None이 됨
            if git_repo_candidate.repo:
                 self.git = git_repo_candidate
            # else: # GitRepo 초기화 실패 메시지는 생성자에서 출력
            #    pass
        except Exception as e: # 혹시 모를 GitRepo 생성자 외 예외
             self.ui.error(f"GitRepo 객체 생성 실패: {e}", traceback.format_exc())
             # self.git은 None으로 유지됨

        # AI 백엔드 초기화
        if backend_choice != "none":
             if available_backends:
                 try:
                     self.ai = AIProvider(backend_name=backend_choice, config={})
                 except Exception as e: # AIProvider 초기화 실패 시
                     # AIProvider에서 이미 오류 메시지 출력됨
                     self.ui.error(f"AIProvider ('{backend_choice}') 설정 실패. AI 기능 비활성화됨.", traceback.format_exc())
                     self.ai = None # AI 사용 불가 상태로 명시
             else:
                 # 사용 가능한 백엔드가 없는데 'none'이 아닌 것을 선택한 경우 (main에서 걸러지지만 안전장치)
                 self.ui.error(f"AI 백엔드 '{backend_choice}' 로드 불가 (사용 가능한 백엔드 없음). AI 기능 비활성화됨.")
                 self.ai = None
        # else: backend_choice == "none" 이면 self.ai는 None으로 유지

    def _ensure_prereqs(self,cmd:str,needs_git:bool,needs_ai:bool):
        """명령 실행 전제 조건 (Git 저장소, AI 백엔드 활성화 여부) 확인"""
        if needs_git and not self.git:
            self.ui.error(f"'{cmd}' 명령은 Git 저장소 내에서 실행해야 합니다."); sys.exit(1)
        if needs_ai and not self.ai:
            self.ui.error(f"'{cmd}' 명령은 AI 백엔드가 설정되어야 합니다. ('--backend' 옵션 확인)"); sys.exit(1)

    def save(self):
        self._ensure_prereqs("save", True, True)
        try:
            # --- 기본 작업 이름 설정 (self.git 확인 추가) ---
            default_task_name = "작업 요약" # 안전한 기본값
            if self.git and self.git.repo: # self.git과 self.git.repo가 모두 유효할 때
                try:
                     current_branch = self.git.get_current_branch()
                     if current_branch and not current_branch.startswith("DETACHED_HEAD"):
                         default_task_name = current_branch
                     # 현재 브랜치 이름이 없거나 DETACHED 라면 마지막 커밋 메시지 사용 시도
                     elif self.git.repo.head.is_valid():
                         default_task_name = self.git.repo.head.commit.summary
                except Exception as e_git_info:
                     print(f"[yellow]경고: 기본 작업 이름 가져오기 실패 ({e_git_info})[/]")
            # --- 기본 작업 이름 설정 끝 ---

            task_name_input = self.ui.task_name(default=default_task_name)

            # --- 컨텍스트 요약 기본값 설정 (self.git 확인 추가) ---
            default_context_summary = "작업 내용을 입력하세요." # Git 없을 시 기본값
            if self.git:
                 last_saved_commit = self.git.get_last_state_commit()
                 default_context_summary = self.git.get_commit_messages_since(last_saved_commit.hexsha if last_saved_commit else None)
            # --- 컨텍스트 요약 기본값 설정 끝 ---

            context_summary_input = self.ui.multiline("작업 내용 요약 (AI가 생성한 요약을 붙여넣거나 직접 작성)", default=default_context_summary)
            if not context_summary_input.strip(): self.ui.error("작업 내용 요약이 비어있어 저장을 취소합니다."); return

            # 아티팩트 목록 생성 (self.art_dir 확인 추가)
            current_artifacts = []
            if self.art_dir.exists() and self.art_dir.is_dir():
                 current_artifacts = [f.name for f in self.art_dir.iterdir() if f.is_file()]
                 self.ui.console.print(f"[dim]현재 아티팩트 ({self.art_dir.relative_to(self.app_root)}): {', '.join(current_artifacts) or '없음'}[/]")
            else:
                 self.ui.console.print(f"[yellow]경고: 아티팩트 폴더({self.art_dir.relative_to(self.app_root)})를 찾을 수 없습니다.[/]")

            self.ui.console.print("\n[bold yellow]AI가 인수인계 문서를 생성 중입니다...[/]")
            # self.ai는 _ensure_prereqs에서 None 아님을 보장
            generated_markdown = self.ai.make_summary(task_name_input, context_summary_input, current_artifacts)
            self.ui.panel(generated_markdown, "AI 생성 요약본 (검증 전)")

            self.ui.console.print("[bold yellow]생성된 요약본을 AI가 검증 중입니다...[/]")
            is_valid_summary, validation_message = self.ai.verify_summary(generated_markdown)
            if not is_valid_summary: raise RuntimeError(f"AI가 생성한 인수인계 문서 검증 실패:\n{validation_message}")
            self.ui.notify("AI 검증 통과!", style="green")

            # Serializer.save_state 호출 (예외 처리 강화됨)
            saved_state_files, artifact_snapshot_dir = Serializer.save_state(generated_markdown, task_name_input, self.state_dir, self.art_dir, self.app_root)

            # Git 저장 (self.git은 _ensure_prereqs에서 None 아님을 보장)
            commit_short_hash = self.git.save(saved_state_files, task_name_input, artifact_snapshot_dir)
            self.ui.notify(f"인수인계 상태 저장 완료! (Commit: {commit_short_hash})", style="bold green")

            # HTML 파일 경로 찾기 (Optional 고려)
            generated_html_file = next((f for f in saved_state_files if f.name.endswith(".html")), None)
            if generated_html_file and generated_html_file.exists():
                 self.ui.console.print(f"[dim]HTML 프리뷰 생성됨: {generated_html_file.relative_to(self.app_root)}[/]")

        except Exception as e: self.ui.error(f"Save 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def load(self, latest: bool = False):
        self._ensure_prereqs("load", True, True)
        try:
            # self.git은 None 아님 보장
            saved_states_list = self.git.list_states(self.state_dir)
            if not saved_states_list: self.ui.error("저장된 인수인계 상태가 없습니다."); return

            selected_commit_hash: Optional[str]
            if latest:
                if not saved_states_list: # list_states 결과가 비어있을 경우 방지
                    self.ui.error("저장된 상태가 없어 최근 상태를 로드할 수 없습니다."); return
                selected_commit_hash = saved_states_list[-1]["hash"]
                print(f"[info]가장 최근 상태 로드 중: {selected_commit_hash} (작업명: {saved_states_list[-1]['task']})[/]")
            else: selected_commit_hash = self.ui.pick_state(saved_states_list)

            if not selected_commit_hash: return # 사용자가 취소

            self.ui.console.print(f"[bold yellow]{selected_commit_hash} 커밋에서 상태 정보를 로드 중입니다...[/]")
            markdown_content = self.git.load_state(selected_commit_hash, self.state_dir)
            prompt_formatted_content = Serializer.to_prompt(markdown_content, selected_commit_hash)
            self.ui.panel(prompt_formatted_content, f"로드된 상태 (Commit: {selected_commit_hash})", border_style="cyan")

            self.ui.console.print("[bold yellow]AI가 로드된 상태를 분석하고 이해도를 보고합니다...[/]")
            # self.ai는 None 아님 보장
            ai_report = self.ai.load_report(markdown_content)
            self.ui.panel(ai_report, "AI 이해도 보고서", border_style="magenta")

        except Exception as e: self.ui.error(f"Load 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def diff(self, target: str = "HEAD"):
        self._ensure_prereqs("diff", True, False) # AI 불필요
        try:
            self.ui.console.print(f"[bold yellow]'{target}' 대비 현재 변경 사항을 확인 중입니다... (Git 추적 파일 기준)[/]")
            # self.git은 None 아님 보장
            diff_output_text = self.git.get_diff(target, color=True)
            self.ui.diff_panel(diff_output_text, target)
        except Exception as e: self.ui.error(f"Diff 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def verify_checksums(self, commit_hash: str):
        self._ensure_prereqs("verify", True, False) # AI 불필요
        self.ui.console.print(f"[dim]커밋 {commit_hash}의 저장된 아티팩트 체크섬 정보를 표시합니다. (실제 파일 비교 검증은 아직 구현되지 않았습니다.)[/]")
        # self.git은 None 아님 보장
        try:
            commit_obj = self.git.repo.commit(self.git.repo.git.rev_parse(commit_hash))
            meta_blob: Optional[Blob] = None
            # Git 저장소 루트 기준 상태 디렉토리 상대 경로
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
        except Exception as e: self.ui.error(f"체크섬 정보 로드/표시 중 오류 ({commit_hash}): {str(e)}", traceback.format_exc())


# --- 스크립트 진입점 ---
def main_cli_entry_point():
    # Git 저장소 루트 또는 현재 작업 디렉토리 결정
    cli_root_path = pathlib.Path('.').resolve() # 기본 CWD
    git_repo_root_obj = None
    is_git_repo_at_cli_root = False
    try:
        git_repo_root_obj = Repo(cli_root_path, search_parent_directories=True)
        cli_root_path = pathlib.Path(git_repo_root_obj.working_tree_dir)
        is_git_repo_at_cli_root = True
    except InvalidGitRepositoryError: pass # Git 저장소 아니면 CWD 사용
    except Exception as e: # 다른 Git 관련 예외 (git 실행파일 못 찾음 등)
         print(f"[yellow]경고: Git 저장소 확인 중 오류 발생 (Git 기능 사용 불가): {e}[/]")

    # 애플리케이션 실행에 필요한 기본 디렉토리 경로 설정 (Handover 클래스 생성 시 전달됨)
    app_state_dir = cli_root_path / "ai_states"
    app_art_dir = cli_root_path / "artifacts"
    # 필요시 생성 시도 (권한 문제 등 있을 수 있음)
    try:
        app_state_dir.mkdir(parents=True, exist_ok=True)
        app_art_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         print(f"[red]오류: 필수 디렉토리 생성 실패 ({app_state_dir} 또는 {app_art_dir}): {e}[/]")
         sys.exit(1)

    # --- Argument Parser 설정 ---
    parser = argparse.ArgumentParser(description="AI 기반 프로젝트 인수인계 상태 관리 도구 (v1.1.6)", formatter_class=argparse.RawTextHelpFormatter)
    backend_choices_list = list(available_backends.keys()) if available_backends else []
    default_be = "none" # AI 기능 없이 실행이 기본
    if "ollama" in backend_choices_list: default_be = "ollama"
    elif backend_choices_list: default_be = backend_choices_list[0]

    parser.add_argument("--backend", default=os.getenv("AI_BACKEND", default_be),
                        choices=backend_choices_list + ["none"], # "none"을 명시적 선택지로
                        help=f"AI 백엔드 (기본값: 환경변수 AI_BACKEND 또는 '{default_be}'). 사용가능: {', '.join(backend_choices_list) or '없음'}. 'none'으로 AI 비활성화.")

    subparsers = parser.add_subparsers(dest="command", help="실행할 작업", required=True)
    # (명령어, 도움말, Git 필요여부, AI 필요여부)
    cmd_configs = [("save", "현재 작업 상태를 요약/저장", True, True),
                   ("load", "과거 저장된 상태 불러오기", True, True),
                   ("diff", "현재 변경 사항 미리보기 (Git)", True, False),
                   ("verify", "저장된 상태 아티팩트 체크섬 표시", True, False)]
    for name, help_txt, git_req, ai_req in cmd_configs:
        p = subparsers.add_parser(name, help=f"{help_txt}{' (Git 필요)' if git_req else ''}{' (AI 필요)' if ai_req else ''}")
        if name == "load": p.add_argument("-l", "--latest", action="store_true", help="가장 최근 상태 자동 로드")
        if name == "diff": p.add_argument("target", nargs="?", default="HEAD", help="비교 대상 Commit/Branch (기본값: HEAD)")
        if name == "verify": p.add_argument("commit", help="체크섬 정보를 확인할 상태 커밋 해시")

    args = parser.parse_args()

    # --- 실행 전 조건 검사 ---
    chosen_cmd_config = next((c for c in cmd_configs if c[0] == args.command), None)
    if not chosen_cmd_config: # 혹시 모를 오류 방지
         UI.error(f"알 수 없는 명령어: {args.command}"); sys.exit(1)

    _, _, git_needed, ai_needed = chosen_cmd_config

    # Git 필요 여부 검사
    if git_needed and not is_git_repo_at_cli_root:
        UI.error(f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 위치는 Git 저장소가 아님: {cli_root_path})"); sys.exit(1)

    # AI 필요 여부 검사
    if ai_needed:
        if args.backend == "none":
             UI.error(f"'{args.command}' 명령 실행 불가: AI 기능이 필요하지만 '--backend' 옵션이 'none'으로 설정되었습니다."); sys.exit(1)
        # 'none'이 아닌데 사용 가능한 백엔드가 없는 경우 (AIBaseBackend 로딩 실패 등)
        elif args.backend not in available_backends:
             UI.error(f"'{args.command}' 명령 실행 불가: 선택된 AI 백엔드 '{args.backend}'를 로드할 수 없습니다. 'backends' 폴더 또는 백엔드 설정을 확인하세요."); sys.exit(1)

    # --- 기본 정보 출력 ---
    print(f"[bold underline]Handover 스크립트 v1.1.6[/]")
    if is_git_repo_at_cli_root: print(f"[dim]프로젝트 루트 (Git): {cli_root_path}[/]")
    else: print(f"[dim]현재 작업 폴더 (Git 저장소 아님): {cli_root_path}[/]")

    # AI 백엔드 사용 정보 출력 (AI가 필요 없거나 'none'이면 출력 안함)
    if args.backend != "none" and args.backend in available_backends:
         print(f"[dim]AI 백엔드 사용: [bold cyan]{args.backend}[/][/dim]")
    elif args.backend == "none" and not ai_needed: # AI 불필요 명령인데 none 선택 시
         print(f"[dim]AI 백엔드: [bold yellow]none (비활성화됨)[/][/dim]")
    # AI 필요한데 none 이거나, none 아닌데 로드 실패한 경우는 위에서 에러 처리됨

    # --- 핸들러 실행 ---
    try:
        # Handover 초기화 (GitRepo, AIProvider 생성 시도)
        handler = Handover(backend_choice=args.backend, current_app_root=cli_root_path)

        # 명령어 실행 (각 메소드에서 _ensure_prereqs 로 최종 확인)
        if args.command == "save": handler.save()
        elif args.command == "load": handler.load(latest=args.latest)
        elif args.command == "diff": handler.diff(target=args.target)
        elif args.command == "verify": handler.verify_checksums(commit_hash=args.commit)
    except Exception as e_handler: # Handover 초기화 또는 메소드 실행 중 예외
        UI.error("핸들러 실행 중 예기치 않은 오류", traceback.format_exc()); sys.exit(1)

if __name__ == "__main__":
    if sys.version_info < (3, 8): print("[bold red]오류: Python 3.8 이상 필요.[/]"); sys.exit(1)
    main_cli_entry_point()