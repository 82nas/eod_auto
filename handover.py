#!/usr/bin/env python3
# handover.py – 인수인계 v1.1.5 (Serializer.save_state 구문 최종 수정 및 경로 처리 개선)

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
from typing import List, Dict, Tuple, Optional, Type, Any
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
except ImportError as e:
    print(f"[bold red]오류: 필요한 라이브러리가 설치되지 않았습니다.[/]\n{e}")
    print("팁: [yellow]pip install gitpython requests rich python-dotenv markdown2[/] 명령을 실행하세요.")
    sys.exit(1)

load_dotenv()

# --- 경로 & 상수 (초기 정의) ---
# 이 값들은 main_cli_entry_point 함수에서 실제 Git 루트 또는 CWD 기준으로 재설정됩니다.
# Serializer 등 다른 모듈에서 직접 참조하는 경우를 위해 모듈 레벨에도 둡니다.
# Handover 클래스는 인스턴스 생성 시 전달받은 current_root를 기준으로 자체 경로(self.state_dir 등)를 사용합니다.
APPLICATION_ROOT_PATH = pathlib.Path('.').resolve() # 스크립트 실행 기준 초기 CWD
# 아래 변수들은 main_cli_entry_point에서 업데이트될 예정
MODULE_STATE_DIR = APPLICATION_ROOT_PATH / "ai_states"
MODULE_ART_DIR = APPLICATION_ROOT_PATH / "artifacts"
BACKEND_DIR = pathlib.Path(__file__).resolve().parent / "backends" # handover.py와 backends 폴더가 같은 위치에 있다고 가정
COMMIT_TAG = "state("

# --- AI 백엔드 로딩 ---
try:
    # backends.base 모듈의 정확한 경로를 사용하여 임포트
    base_spec = importlib.util.spec_from_file_location("backends.base", BACKEND_DIR / "base.py")
    if base_spec is None or base_spec.loader is None:
        raise ImportError(f"backends.base 모듈 스펙을 찾을 수 없습니다. 경로: {BACKEND_DIR / 'base.py'}")
    backends_base_module = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(backends_base_module) # 모듈 실행
    AIBaseBackend = backends_base_module.AIBaseBackend # AIBaseBackend 클래스 가져오기
except ImportError as e:
    print(f"[bold red]오류: backends.base 모듈 임포트 실패: {e}[/]")
    sys.exit(1)
except AttributeError: # AIBaseBackend 클래스를 찾지 못한 경우
    print(f"[bold red]오류: backends.base 모듈에서 AIBaseBackend 클래스를 찾을 수 없습니다.[/]")
    sys.exit(1)
except FileNotFoundError:
    print(f"[bold red]오류: backends/base.py 파일을 찾을 수 없습니다. 경로: {BACKEND_DIR / 'base.py'}[/]")
    sys.exit(1)


available_backends: Dict[str, Type[AIBaseBackend]] = {}
if BACKEND_DIR.exists() and BACKEND_DIR.is_dir():
    for f_py in BACKEND_DIR.glob("*.py"):
        module_name_stem = f_py.stem
        if module_name_stem == "__init__" or module_name_stem == "base":
            continue
        try:
            # 각 백엔드 모듈 임포트
            spec = importlib.util.spec_from_file_location(f"backends.{module_name_stem}", f_py)
            if spec is None or spec.loader is None: continue # 스펙 로드 실패 시 건너뛰기
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module) # 모듈 실행
            
            for name, obj in module.__dict__.items(): # 모듈의 모든 속성 검사
                if (isinstance(obj, type) and          # 클래스인지 확인
                        issubclass(obj, AIBaseBackend) and # AIBaseBackend의 하위 클래스인지 확인
                        obj is not AIBaseBackend):         # AIBaseBackend 자체는 아닌지 확인
                    # get_name이 staticmethod로 정의되어 클래스에서 직접 호출 가능해야 함
                    backend_name_from_class = obj.get_name() 
                    if backend_name_from_class != "base": # "base"는 실제 백엔드 이름이 아님
                        available_backends[backend_name_from_class] = obj
        except ImportError as e:
            print(f"[yellow]경고: 백엔드 모듈 '{module_name_stem}' 로딩 실패: {e}[/]")
        except AttributeError as e: # get_name 메소드가 없거나 staticmethod가 아닌 경우 등
             print(f"[yellow]경고: 백엔드 클래스 '{name if 'name' in locals() else module_name_stem}' 속성 오류: {e}[/]")
        except Exception as e: # 그 외 예기치 않은 오류
            print(f"[yellow]경고: 백엔드 파일 '{f_py.name}' 처리 중 예외 발생: {e}[/]")
else:
    # 백엔드 디렉토리가 없는 경우 (스크립트 설정 오류 가능성)
    print(f"[yellow]경고: 백엔드 디렉토리 '{BACKEND_DIR}'를 찾을 수 없거나 디렉토리가 아닙니다.[/]")


# --- AIProvider 클래스 ---
class AIProvider:
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        if not available_backends and backend_name != "none": # "none"이 아니고 사용 가능한 백엔드가 없을 때
             raise RuntimeError("AIProvider 초기화 실패: 사용 가능한 AI 백엔드가 없습니다.")
        
        if backend_name == "none": # AI 기능 비활성화 선택
            self.backend = None 
            print(f"[dim]AI 백엔드: [bold yellow]none (비활성화됨)[/][/dim]")
            return

        if backend_name not in available_backends: # 선택한 백엔드가 목록에 없을 때
            raise ValueError(f"알 수 없는 백엔드: '{backend_name}'. 사용 가능: {list(available_backends.keys()) + ['none']}")
        
        BackendClass = available_backends[backend_name]
        try:
            self.backend: Optional[AIBaseBackend] = BackendClass(config) # 타입 힌트 명시
            print(f"[dim]AI 백엔드 사용: [bold cyan]{backend_name}[/][/dim]")
        except Exception as e: # 백엔드 클래스 초기화 중 예외 발생 시
            print(f"[bold red]오류: 백엔드 '{backend_name}' 초기화 실패: {e}[/]")
            if hasattr(BackendClass, 'get_config_description'): # 설정 정보가 있다면 출력
                print(f"[yellow]필요 설정:\n{BackendClass.get_config_description()}[/]")
            raise e # 예외 다시 발생시켜 프로그램 중단 또는 상위에서 처리

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
                  # 나머지 헤더는 '## 헤더명' 형태
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
        try: 
            self.repo = Repo(repo_path)
            if self.repo.bare: # Bare 저장소는 작업 디렉토리가 없으므로 지원 안함
                 raise InvalidGitRepositoryError(f"'{repo_path}'는 bare 저장소입니다. 작업 디렉토리가 있는 저장소가 필요합니다.")
        except InvalidGitRepositoryError: # Git 저장소가 아닌 경우
            raise # 예외를 다시 발생시켜 main_cli_entry_point에서 처리하도록 함
    
    def _safe(self, git_func, *args, **kwargs): # Git 명령어 안전 실행 래퍼
        try: return git_func(*args, **kwargs)
        except GitCommandError as e: stderr = e.stderr.strip(); raise RuntimeError(f"Git 명령어 실패: {e.command}\n오류: {stderr}") from e

    def get_last_state_commit(self) -> Optional[Commit]:
         try:
             for c in self.repo.iter_commits(max_count=200, first_parent=True): # 히스토리 복잡도 고려
                 if c.message.startswith(COMMIT_TAG): return c
         except Exception: pass # 오류 발생 시 None 반환
         return None

    def get_commit_messages_since(self, commit_hash: Optional[str]) -> str:
        if not self.repo: return "Git 저장소가 초기화되지 않았습니다."
        if not commit_hash: # 이전 상태 커밋이 없는 경우 (최초 실행 등)
            try: 
                commits = list(self.repo.iter_commits(max_count=10, no_merges=True)) # 최근 10개 (머지 커밋 제외)
                log = "\n".join(f"- {c.hexsha[:7]}: {c.summary}" for c in reversed(commits)) # 시간순 정렬
                return f"최근 커밋 {len(commits)}개:\n{log}" if log else "최근 커밋 없음"
            except Exception as e: return f"최근 커밋 로그 조회 실패: {e}"
        try: 
            self.repo.commit(commit_hash) # 주어진 해시가 유효한 커밋인지 확인
            log_cmd = f"{commit_hash}..HEAD" # commit_hash 다음부터 HEAD까지
            commit_log = self.repo.git.log(log_cmd, '--pretty=format:- %h: %s', '--abbrev-commit', '--no-merges')
            return f"'{commit_hash[:8]}' 이후 커밋:\n{commit_log}" if commit_log else f"'{commit_hash[:8]}' 이후 커밋 없음"
        except GitCommandError as e: # Git 명령 자체의 오류
            return f"커밋 로그 조회 실패 ({commit_hash}): {e.stderr}"
        except Exception as e: # 그 외 (예: 존재하지 않는 커밋 해시)
            return f"커밋 로그 조회 중 오류 ({commit_hash}): {e}"

    def get_current_branch(self) -> Optional[str]:
        if not self.repo: return "Git 저장소 없음"
        try: return self.repo.active_branch.name
        except TypeError: # Detached HEAD 상태일 때 active_branch가 오류 발생
            try: return f"DETACHED_HEAD@{self.repo.head.commit.hexsha[:7]}" # 현재 커밋 해시로 표시
            except Exception: return "DETACHED_HEAD" # 그것도 실패하면 단순 문자열
        except Exception: return None # 그 외 브랜치 조회 실패

    def get_diff(self, target: str = "HEAD", color: bool = True) -> str:
        if not self.repo: return "Git 저장소가 초기화되지 않았습니다."
        try:
            self.repo.commit(target) # target이 유효한 커밋/브랜치/태그인지 확인
            color_opt = '--color=always' if color else '--color=never'
            
            # Staged changes (Index vs Target Commit)
            staged_diff = self.repo.git.diff('--staged', target, color_opt)
            # Unstaged changes (Working Directory vs Target Commit) - Staged 내용은 제외하고 보여주려면 Index와 비교해야 하나,
            # 사용자는 보통 최종 커밋(Target) 대비 전체 변경을 보고 싶어함.
            # 그래서 Target 대비 Working Directory 전체를 보여주는 것이 더 직관적일 수 있음.
            # 다만, 이러면 Staged된 내용이 두 번 나올 수 있으므로, Staged와 Unstaged(WD vs Index)로 구분하는게 나을수도.
            # 여기서는 (WD vs Target)과 (Index vs Target)을 보여주기로 함.
            working_tree_vs_target_diff = self.repo.git.diff(target, color_opt)

            diff_output = ""
            has_staged = bool(staged_diff.strip())
            # working_tree_vs_target_diff가 staged_diff와 다를 때만 unstaged로 간주 (중복 방지)
            has_meaningful_wt_diff = bool(working_tree_vs_target_diff.strip()) and (working_tree_vs_target_diff != staged_diff)

            if has_staged:
                diff_output += f"--- Staged Changes (vs {target}) ---\n{staged_diff}\n\n"
            
            if has_meaningful_wt_diff:
                 diff_output += f"--- Unstaged Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff if not has_staged else self.repo.git.diff(color_opt)}\n" # If staged, show WD vs Index
            elif not has_staged and bool(working_tree_vs_target_diff.strip()): # Only working tree changes vs target
                 diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"
            
            return diff_output.strip() if diff_output.strip() else f"'{target}'과(와) 변경 사항 없음 (Git 추적 파일 기준)"
        except GitCommandError as e: return f"Diff 생성 Git 오류: {e.stderr}"
        except Exception as e: return f"Diff 생성 오류: {e}"

    def save(self, state_paths: List[pathlib.Path], task: str, snapshot_dir: Optional[pathlib.Path]) -> str:
        if not self.repo: raise RuntimeError("Git 저장소가 없어 저장할 수 없습니다.")
        paths_to_add_str = [str(p.resolve()) for p in state_paths]
        if snapshot_dir and snapshot_dir.exists() and any(snapshot_dir.iterdir()): 
            paths_to_add_str.append(str(snapshot_dir.resolve()))
        
        if not state_paths: raise ValueError("저장할 상태 파일(.md, .meta.json)이 없습니다.")
        
        self._safe(self.repo.git.add, *paths_to_add_str)
        commit_msg = f"{COMMIT_TAG}{task})"
        try: 
            self._safe(self.repo.index.commit, commit_msg)
        except RuntimeError as e:
            # "nothing to commit" 또는 유사 메시지 확인 (Git 버전에 따라 다를 수 있음)
            if "nothing to commit" in str(e).lower() or \
               "no changes added to commit" in str(e).lower() or \
               "changes not staged for commit" in str(e).lower() : # 좀 더 관대하게
                print("[yellow]경고: 커밋할 변경 사항이 없습니다. 이전 상태와 동일할 수 있습니다.[/]")
                return self.repo.head.commit.hexsha[:8] + " (변경 없음)"
            raise e # 그 외 커밋 오류는 다시 발생시킴
        
        # Push 로직 수정
        if self.repo.remotes: # 원격 저장소가 설정되어 있는지 확인
            try: 
                current_branch_name = self.get_current_branch()
                if current_branch_name and not current_branch_name.startswith("DETACHED_HEAD"):
                    # 현재 브랜치를 origin으로 푸시 (upstream 설정 가정)
                    self._safe(self.repo.git.push, 'origin', current_branch_name)
                    print("[green]원격 저장소에 푸시 완료.[/]")
                else:
                    print(f"[yellow]경고: 현재 브랜치({current_branch_name})가 특정되지 않았거나 Detached HEAD 상태이므로 푸시를 건너뜁니다.[/]")
            except RuntimeError as e: # _safe에서 발생시킨 RuntimeError (GitCommandError 포함)
                print(f"[yellow]경고: 원격 저장소 푸시 실패. 로컬에는 커밋되었습니다. ({e})[/]")
            except Exception as e_general: # 그 외 예외
                print(f"[yellow]경고: 원격 저장소 푸시 중 예기치 않은 오류: {e_general}[/]")
        else: 
            print("[yellow]경고: 설정된 원격 저장소가 없어 푸시를 건너뜁니다.[/]")
        
        return self.repo.head.commit.hexsha[:8]

    def list_states(self, current_app_state_dir: pathlib.Path) -> List[Dict]:
        if not self.repo: return []
        items = []
        search_path_str = str(current_app_state_dir.relative_to(self.repo.working_dir)) if current_app_state_dir.is_absolute() else str(current_app_state_dir)

        try: 
            # 특정 경로(ai_states)의 변경사항이 있는 커밋만 조회 시도
            commits = list(self.repo.iter_commits(max_count=100, first_parent=True, paths=search_path_str))
        except Exception: # 경로 필터링 실패 시 (예: 경로에 아직 아무것도 없을 때) 모든 커밋 조회로 fallback
            commits = list(self.repo.iter_commits(max_count=100, first_parent=True))
        
        for c in commits:
            if not c.message.startswith(COMMIT_TAG): continue
            headline = ""; meta_blob = None
            try:
                # 커밋 트리 내에서 current_app_state_dir 이름으로 시작하는 경로에서 .meta.json 찾기
                # current_app_state_dir.name은 'ai_states'
                # item.path는 'ai_states/filename.meta.json' 형태
                found_meta = False
                if current_app_state_dir.name in c.tree: # 커밋 트리에 'ai_states' 폴더가 있는지 확인
                    for item in c.tree[current_app_state_dir.name].traverse(): # 'ai_states' 폴더 내부만 탐색
                        if isinstance(item, Blob) and item.name.endswith(".meta.json"): # Blob의 이름이 .meta.json으로 끝나는지
                            meta_blob = item; found_meta = True; break
                if not found_meta: # 전체 트리에서 경로명으로 재탐색 (더 안전)
                    for item in c.tree.traverse():
                         if isinstance(item, Blob) and item.path.startswith(current_app_state_dir.name) and item.path.endswith(".meta.json"):
                             meta_blob = item; break
                
                if meta_blob: 
                    metadata = json.loads(meta_blob.data_stream.read().decode('utf-8'))
                    headline = metadata.get("headline", "")
            except Exception: headline = "[메타데이터 오류]"
            items.append({"hash": c.hexsha[:8], "task": c.message[len(COMMIT_TAG):-1].strip(), "time": datetime.datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d %H:%M"), "head": headline or "-"})
        return list(reversed(items))

    def load_state(self, commit_hash: str, current_app_state_dir: pathlib.Path) -> str:
        if not self.repo: raise RuntimeError("Git 저장소가 없어 로드할 수 없습니다.")
        try: commit_obj = self.repo.commit(self.repo.git.rev_parse(commit_hash)) # 특정 커밋 가져오기
        except Exception as e: raise RuntimeError(f"커밋 '{commit_hash}' 접근 오류: {e}") from e
        
        try:
            # 커밋 트리 내에서 current_app_state_dir 이름으로 시작하는 경로에서 .md 파일 찾기
            commit_state_dir_name_str = current_app_state_dir.name
            if commit_state_dir_name_str in commit_obj.tree: # 커밋 트리에 'ai_states' 폴더가 있는지 확인
                for item in commit_obj.tree[commit_state_dir_name_str].traverse(): # 'ai_states' 폴더 내부만 탐색
                    if isinstance(item, Blob) and item.name.endswith(".md"): # Blob의 이름이 .md로 끝나는지
                        return item.data_stream.read().decode('utf-8')
            # Fallback: 전체 트리에서 경로명으로 재탐색
            for item in commit_obj.tree.traverse():
                 if isinstance(item, Blob) and item.path.startswith(commit_state_dir_name_str) and item.path.endswith(".md"):
                     return item.data_stream.read().decode('utf-8')
            raise RuntimeError(f"커밋 '{commit_hash}' 내 '{commit_state_dir_name_str}' 폴더에서 상태 파일(.md)을 찾을 수 없습니다.")
        except KeyError: # commit_obj.tree에 commit_state_dir_name_str 키가 없는 경우
            raise RuntimeError(f"커밋 '{commit_hash}'에 '{current_app_state_dir.name}' 폴더가 없습니다. 이전 버전의 상태일 수 있습니다.")
        except Exception as e: # 그 외 파일 읽기 오류 등
            raise RuntimeError(f"커밋 '{commit_hash}' 상태 로드 중 예기치 않은 오류: {e}")

    # --- Serializer 클래스 ---
    class Serializer:
        @staticmethod
        def _calculate_sha256(fp: pathlib.Path) -> Optional[str]:
            h = hashlib.sha256()
            try:
                with open(fp, "rb") as f:
                    while True: b = f.read(4096);
                    if not b: break; h.update(b)
                return h.hexdigest()
            except IOError: return None # 파일 읽기 오류 시 None 반환

        @staticmethod
        def _generate_html(md: str, title: str) -> str:
             css = """<style>body{font-family:sans-serif;line-height:1.6;padding:20px;max-width:800px;margin:auto;color:#333}h1,h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.5em;margin-bottom:1em}h1{font-size:2em}h2{font-size:1.5em}ul,ol{padding-left:2em}li{margin-bottom:.5em}code{background-color:#f0f0f0;padding:.2em .4em;border-radius:3px;font-family:monospace;font-size:.9em}pre{background-color:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}pre code{background-color:transparent;padding:0;border-radius:0}blockquote{border-left:4px solid #ccc;padding-left:1em;color:#666;margin-left:0}table{border-collapse:collapse;width:100%;margin-bottom:1em}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}</style>"""
             body = markdown2.markdown(md, extras=["metadata","fenced-code-blocks","tables","strike","task_list","code-friendly","markdown-in-html"])
             title_meta = title;
             if hasattr(body,"metadata") and body.metadata.get("title"): title_meta = body.metadata["title"]
             return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{textwrap.shorten(title_meta, width=50, placeholder="...")}</title>{css}</head><body>{body}</body></html>"""
        
        @staticmethod
        def save_state(md: str, task: str, 
                       # 이 메소드는 이제 Handover 인스턴스에서 올바른 경로를 전달받음
                       current_app_state_dir: pathlib.Path, 
                       current_app_art_dir: pathlib.Path, 
                       current_app_root: pathlib.Path
                       ) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
            
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            safe_task_name = "".join(c for c in task if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
            if not safe_task_name: safe_task_name = "untitled_task" # 태스크 이름이 안전하지 않은 문자만 있을 경우 대비
            base_fn = f"{ts}_{safe_task_name}"
            
            # 전달받은 경로 사용
            current_app_state_dir.mkdir(exist_ok=True)
            current_app_art_dir.mkdir(exist_ok=True)

            state_f = current_app_state_dir / f"{base_fn}.md"
            html_f = current_app_state_dir / f"{base_fn}.html"
            meta_f = current_app_state_dir / f"{base_fn}.meta.json"
            
            try: state_f.write_text(md, encoding="utf-8")
            except IOError as e: raise RuntimeError(f"MD 파일 저장 실패 ({state_f}): {e}") from e
            
            html_ok = False
            try: 
                html_content = Serializer._generate_html(md,task)
                html_f.write_text(html_content,encoding="utf-8")
                html_ok=True
            except Exception as e: print(f"[yellow]경고: HTML 프리뷰 생성/저장 실패 ({html_f.name}): {e}[/]")
            
            snap_dir = None; checksums = {}
            arts = [f for f in current_app_art_dir.iterdir() if f.is_file()] if current_app_art_dir.exists() else []
            
            if arts:
                # 스냅샷 디렉토리 이름을 base_fn 기준으로 생성 (타임스탬프 + 태스크명)
                snapshot_sub_dir_name = f"{base_fn}_artifacts"
                snap_dir = current_app_art_dir / snapshot_sub_dir_name 
                snap_dir.mkdir(parents=True,exist_ok=True)
                
                print(f"[dim]아티팩트 스냅샷 ({len(arts)}개) -> '{snap_dir.relative_to(current_app_root)}'[/]")
                for f_art in arts:
                    try: 
                        target_path = snap_dir / f_art.name # 스냅샷 폴더 안에 원본 파일명으로 복사
                        shutil.copy2(f_art, target_path)
                        cs = Serializer._calculate_sha256(target_path)
                        if cs: 
                            checksums[f_art.name] = cs # 파일명:해시 형태로 저장
                    except Exception as copy_e: 
                        print(f"[yellow]경고: 아티팩트 파일 '{f_art.name}' 복사/해시 실패: {copy_e}[/]")
            else:
                print("[dim]저장할 아티팩트 파일이 없습니다.[/]")
            
            # 헤드라인 추출 로직 개선
            headline = task # 기본값은 태스크 이름
            for ln in md.splitlines(): 
                ln_s = ln.strip()
                if ln_s.startswith("#"): # 첫번째 # 또는 ## 헤더를 헤드라인으로
                    headline=ln_s.lstrip('# ').strip()
                    break 
            
            meta = {"task":task,"ts":ts,"headline":headline,"artifact_checksums":checksums}
            try: meta_f.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
            except IOError as e: raise RuntimeError(f"메타데이터 파일 저장 실패 ({meta_f}): {e}") from e
            
            paths_to_commit = [state_f, meta_f]
            if html_ok and html_f.exists(): paths_to_commit.append(html_f)
            
            # 실제 파일이 있는 스냅샷 디렉토리만 반환
            valid_snap_dir = snap_dir if (snap_dir and snap_dir.exists() and any(snap_dir.iterdir())) else None
            return paths_to_commit, valid_snap_dir
        
        @staticmethod
        def to_prompt(md: str, commit: str) -> str: 
            return f"### 이전 상태 (Commit: {commit}) ###\n\n{md}\n\n### 상태 정보 끝 ###"

    # --- UI 클래스 ---
    class UI:
        console = Console() # 콘솔 객체 공유
        @staticmethod
        def task_name(default:str="작업 요약") -> str: 
            return Prompt.ask("[bold cyan]작업 이름[/]",default=default)
        @staticmethod
        def multiline(label: str, default: str = "") -> str:
            UI.console.print(f"\n[bold cyan]{label}[/]")
            UI.console.print("[dim](입력을 마치려면 빈 줄에서 Enter 키를 두 번 누르세요)[/]")
            lines = []
            if default: 
                # 기본값이 너무 길면 일부만 보여주거나, Panel 높이 제한
                default_preview = textwrap.shorten(default, width=100, placeholder="...") if len(default.splitlines()) > 5 else default
                print(Panel(default_preview,title="[dim]자동 제안된 내용 (편집 가능, 전체 내용은 아래 입력)[/]",border_style="dim",expand=False))
            
            blank_count = 0
            while True:
                try: line = input() # 표준 input 사용
                except EOFError: break # Ctrl+D/Ctrl+Z 입력 시 종료
                if line=="": 
                    blank_count+=1
                    if blank_count>=2: break # 빈 줄 두 번이면 입력 종료
                else: 
                    blank_count=0; lines.append(line)
            
            final_text = "\n".join(lines).strip()
            if not final_text and default: # 사용자가 아무것도 입력 안하고 기본값이 있었으면 기본값 사용
                 print("[dim]입력 내용이 없어 자동 제안된 내용으로 진행합니다.[/]")
                 return default
            return final_text

        @staticmethod
        def notify(msg:str,style:str="green"): UI.console.print(f"\n[bold {style}]✔ {msg}[/]")
        @staticmethod
        def error(msg:str,details:Optional[str]=None):
            UI.console.print(f"\n[bold red]❌ 오류: {msg}[/]")
            if details: 
                # 상세 오류가 너무 길면 요약 또는 스크롤 가능한 형태로 표시 (Rich 기능 활용)
                UI.console.print(Panel(details,title="[dim]상세 정보 (Traceback)[/]",border_style="dim red",expand=False, height=15 if len(details.splitlines()) > 15 else None))

        @staticmethod
        def pick_state(states:List[Dict])->Optional[str]:
            if not states: print("[yellow]저장된 상태가 없습니다.[/]"); return None
            
            tb = Table(title="[bold]저장된 인수인계 상태 목록[/]",box=box.ROUNDED,show_lines=True, expand=False)
            tb.add_column("#",style="dim",justify="right", width=3)
            tb.add_column("커밋", style="cyan", no_wrap=True, width=10)
            tb.add_column("작업", style="magenta", min_width=20, overflow="fold")
            tb.add_column("시각", style="green", no_wrap=True, width=18)
            tb.add_column("헤드라인", style="yellow", overflow="fold", min_width=30)
            for i,s in enumerate(states):tb.add_row(str(i),s["hash"],s["task"],s["time"],s["head"])
            UI.console.print(tb)
            
            choices=[str(i) for i in range(len(states))]
            selection=Prompt.ask("[bold cyan]로드할 상태의 번호 입력 (취소하려면 Enter)[/]",choices=choices+[""],default="",show_choices=False) # show_choices False로 불필요한 목록 반복 방지
            
            if selection.isdigit() and 0 <= int(selection) < len(states):
                selected_hash=states[int(selection)]["hash"]
                print(f"[info]선택된 커밋: {selected_hash} (작업: {states[int(selection)]['task']})[/]")
                return selected_hash
            
            print("[info]상태 로드를 취소했습니다.[/]")
            return None

        @staticmethod
        def panel(txt:str,title:str,border_style:str="blue"): 
            UI.console.print(Panel(txt,title=f"[bold]{title}[/]",border_style=border_style,expand=False,padding=(1,2)))

        @staticmethod
        def diff_panel(diff_text:str,target:str):
            if not diff_text.strip() or "변경 사항 없음" in diff_text or diff_text.startswith("Diff 생성 오류"): 
                print(f"[dim]{diff_text}[/]") # 오류나 "변경 없음" 메시지는 일반 텍스트로
                return
            # Diff 텍스트가 실제 diff 내용일 때만 Syntax 사용
            syntax_obj = Syntax(diff_text,"diff",theme="default",line_numbers=False,word_wrap=False)
            UI.console.print(Panel(syntax_obj,title=f"[bold]Diff (vs {target})[/]",border_style="yellow",expand=True))

    # --- Handover 클래스 ---
    class Handover:
        def __init__(self, backend_choice: str, current_app_root: pathlib.Path):
            self.ui = UI()
            self.app_root = current_app_root # 스크립트 실행 기준 루트 경로
            self.state_dir = self.app_root / "ai_states"
            self.art_dir = self.app_root / "artifacts"
            
            try: 
                self.git = GitRepo(self.app_root) # GitRepo는 app_root 기준으로 생성
            except InvalidGitRepositoryError: # Git 저장소가 아닌 경우
                self.git = None # Git 기능 사용 불가로 표시
            except Exception as e: # 그 외 GitRepo 초기화 실패
                self.ui.error(f"GitRepo 초기화 실패 ({self.app_root}): {e}", traceback.format_exc())
                sys.exit(1) # 치명적 오류로 간주

            # AIProvider 초기화
            if backend_choice != "none" and available_backends:
                try: self.ai = AIProvider(backend_name=backend_choice, config={})
                except Exception as e: self.ui.error(f"AI 백엔드 ('{backend_choice}') 초기화 실패.", traceback.format_exc()); sys.exit(1)
            elif backend_choice != "none" and not available_backends: # 백엔드 선택했으나 로드된게 없을때
                self.ui.error(f"선택된 AI 백엔드 '{backend_choice}'를 위한 모듈을 찾을 수 없거나 로드 중 오류 발생."); sys.exit(1)
            else: # AI 기능을 사용 안함 ("none" 선택 또는 사용가능 백엔드 없음)
                self.ai = None 
                if backend_choice != "none": # 사용가능 백엔드가 없어서 none으로 된 경우
                     self.ui.console.print("[yellow]경고: 사용 가능한 AI 백엔드가 없어 AI 기능이 비활성화됩니다.[/]")
                # else: 사용자가 명시적으로 'none' 선택한 경우는 AIProvider에서 메시지 출력

        def _ensure_prerequisites(self,command_name:str, git_required:bool, ai_required:bool):
            """명령 실행 전 Git 및 AI 준비 상태 확인"""
            if git_required and not self.git:
                self.ui.error(f"'{command_name}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 루트: {self.app_root})")
                sys.exit(1)
            if ai_required and not self.ai:
                self.ui.error(f"'{command_name}' 명령은 AI 백엔드가 설정되어야 합니다. ('--backend' 옵션을 확인하거나 'none'이 아닌지 확인하세요.)")
                sys.exit(1)

        def save(self):
            self._ensure_prerequisites("save", True, True) # save는 Git과 AI 모두 필요
            try:
                default_task_name = self.git.get_current_branch() or \
                                    (self.git.repo.head.commit.summary if self.git.repo and self.git.repo.head.is_valid() else "작업 요약")
                task_name_input = self.ui.task_name(default=default_task_name)
                
                last_saved_commit = self.git.get_last_state_commit()
                default_context_summary = self.git.get_commit_messages_since(last_saved_commit.hexsha if last_saved_commit else None)
                context_summary_input = self.ui.multiline("작업 내용 요약 (AI가 생성한 요약을 붙여넣거나 직접 작성)", default=default_context_summary)
                
                if not context_summary_input.strip(): 
                    self.ui.error("작업 내용 요약이 비어있습니다. 저장을 취소합니다.")
                    return

                # 디렉토리 존재 보장 (Handover 인스턴스 경로 기준)
                self.state_dir.mkdir(exist_ok=True)
                self.art_dir.mkdir(exist_ok=True)
                
                current_artifacts = [f.name for f in self.art_dir.iterdir() if f.is_file()] if self.art_dir.exists() else []
                if self.art_dir.exists():
                    self.ui.console.print(f"[dim]현재 아티팩트 ({self.art_dir.relative_to(self.app_root)}): {', '.join(current_artifacts) or '없음'}[/]")
                else: # ART_DIR이 어떤 이유로든 없을 경우 (일반적이지 않음)
                    self.ui.console.print(f"[dim]아티팩트 폴더({self.art_dir.relative_to(self.app_root)})가 존재하지 않습니다.[/]")
                
                self.ui.console.print("\n[bold yellow]AI가 인수인계 문서를 생성 중입니다...[/]")
                generated_markdown = self.ai.make_summary(task_name_input, context_summary_input, current_artifacts)
                self.ui.panel(generated_markdown, "AI 생성 요약본 (검증 전)")
                
                self.ui.console.print("[bold yellow]생성된 요약본을 AI가 검증 중입니다...[/]")
                is_valid_summary, validation_message = self.ai.verify_summary(generated_markdown)
                if not is_valid_summary: 
                    raise RuntimeError(f"AI가 생성한 인수인계 문서 검증 실패:\n{validation_message}")
                self.ui.notify("AI 검증 통과!", style="green")
                
                # Serializer에 정확한 경로 전달
                saved_state_files, artifact_snapshot_dir = Serializer.save_state(
                    generated_markdown, task_name_input, 
                    self.state_dir, self.art_dir, self.app_root # Handover 인스턴스의 경로 사용
                )
                
                commit_short_hash = self.git.save(saved_state_files, task_name_input, artifact_snapshot_dir)
                self.ui.notify(f"인수인계 상태 저장 완료! (Commit: {commit_short_hash})", style="bold green")

                generated_html_file = next((f for f in saved_state_files if f.name.endswith(".html")), None)
                if generated_html_file and generated_html_file.exists():
                     self.ui.console.print(f"[dim]HTML 프리뷰 생성됨: {generated_html_file.relative_to(self.app_root)}[/]")

            except Exception as e: 
                self.ui.error(f"Save 작업 중 오류 발생: {str(e)}", traceback.format_exc())

        def load(self, latest: bool = False):
            self._ensure_prerequisites("load", True, True) # load는 Git과 AI 모두 필요
            try:
                saved_states_list = self.git.list_states(self.state_dir) # Handover 인스턴스의 state_dir 사용
                if not saved_states_list: 
                    self.ui.error("저장된 인수인계 상태가 없습니다.")
                    return
                
                selected_commit_hash: Optional[str]
                if latest:
                    selected_commit_hash = saved_states_list[-1]["hash"] # 마지막 항목이 최신
                    print(f"[info]가장 최근 상태 로드 중: {selected_commit_hash} (작업명: {saved_states_list[-1]['task']})[/]")
                else:
                    selected_commit_hash = self.ui.pick_state(saved_states_list)

                if not selected_commit_hash: return # 사용자가 선택 취소

                self.ui.console.print(f"[bold yellow]{selected_commit_hash} 커밋에서 상태 정보를 로드 중입니다...[/]")
                markdown_content = self.git.load_state(selected_commit_hash, self.state_dir) # Handover 인스턴스의 state_dir 사용
                
                prompt_formatted_content = Serializer.to_prompt(markdown_content, selected_commit_hash)
                self.ui.panel(prompt_formatted_content, f"로드된 상태 (Commit: {selected_commit_hash})", border_style="cyan")

                self.ui.console.print("[bold yellow]AI가 로드된 상태를 분석하고 이해도를 보고합니다...[/]")
                ai_report = self.ai.load_report(markdown_content)
                self.ui.panel(ai_report, "AI 이해도 보고서", border_style="magenta")

            except Exception as e: 
                self.ui.error(f"Load 작업 중 오류 발생: {str(e)}", traceback.format_exc())

        def diff(self, target: str = "HEAD"):
            self._ensure_prerequisites("diff", True, False) # diff는 Git만 필요
            try:
                 self.ui.console.print(f"[bold yellow]'{target}' 대비 현재 변경 사항을 확인 중입니다... (Git 추적 파일 기준)[/]")
                 diff_output_text = self.git.get_diff(target, color=True)
                 self.ui.diff_panel(diff_output_text, target)
            except Exception as e: 
                self.ui.error(f"Diff 작업 중 오류 발생: {str(e)}", traceback.format_exc())

        def verify_checksums(self, commit_hash: str):
            self._ensure_prerequisites("verify", True, False) # verify는 Git만 필요
            self.ui.console.print(f"[dim]커밋 {commit_hash}의 저장된 아티팩트 체크섬 정보를 표시합니다. (실제 파일 비교 검증은 아직 구현되지 않았습니다.)[/]")
            try:
                # rev_parse를 통해 정확한 커밋 객체 가져오기
                commit_obj = self.git.repo.commit(self.git.repo.git.rev_parse(commit_hash))
                
                meta_blob: Optional[Blob] = None
                # Handover 인스턴스의 state_dir을 기준으로 커밋 내 경로 생성
                # (self.app_root가 Git 저장소 루트이므로, self.state_dir.name만 사용해도 됨)
                meta_file_path_prefix_in_commit = self.state_dir.name 

                # 커밋 트리에서 메타 파일 탐색
                for item_in_tree in commit_obj.tree.traverse():
                    if isinstance(item_in_tree, Blob) and \
                       item_in_tree.path.startswith(meta_file_path_prefix_in_commit) and \
                       item_in_tree.path.endswith(".meta.json"):
                        meta_blob = item_in_tree
                        break # 첫 번째 찾은 메타 파일 사용
                
                if not meta_blob: 
                    self.ui.error(f"커밋 {commit_hash}에서 메타데이터 파일(.meta.json)을 찾을 수 없습니다. (탐색 경로: {meta_file_path_prefix_in_commit})")
                    return
                
                metadata_content = json.loads(meta_blob.data_stream.read().decode('utf-8'))
                artifact_checksums_data = metadata_content.get("artifact_checksums", {})
                
                if artifact_checksums_data:
                    checksums_pretty_str = json.dumps(artifact_checksums_data, indent=2, ensure_ascii=False)
                    self.ui.panel(checksums_pretty_str, f"저장된 아티팩트 체크섬 (Commit: {commit_hash})", border_style="magenta")
                else: 
                    print(f"[dim]커밋 {commit_hash}에 저장된 아티팩트 체크섬 정보가 없습니다.[/]")

            except GitCommandError: # rev_parse 실패 등
                self.ui.error(f"유효한 커밋 해시가 아니거나 찾을 수 없습니다: '{commit_hash}'.")
            except Exception as e: 
                self.ui.error(f"체크섬 정보 로드/표시 중 오류 ({commit_hash}): {str(e)}", traceback.format_exc())

    # --- 스크립트 진입점 ---
    def main_application_entry():
        global APPLICATION_ROOT_PATH, MODULE_STATE_DIR, MODULE_ART_DIR # 명시적 전역 변수 사용
        
        # 실제 Git 저장소 루트를 찾거나 현재 작업 디렉토리(CWD)를 루트로 사용
        current_process_root = pathlib.Path('.').resolve() # 기본값은 CWD
        is_actually_in_git_repo = False
        try:
            # search_parent_directories=True로 상위 폴더까지 .git 검색
            git_repo_obj = Repo('.', search_parent_directories=True)
            current_process_root = pathlib.Path(git_repo_obj.working_tree_dir) # 실제 Git 루트
            is_actually_in_git_repo = True
        except InvalidGitRepositoryError:
            # Git 저장소가 아니면 current_process_root는 CWD 유지
            pass 
        
        # 모듈 레벨 경로 변수 업데이트 (주로 Serializer 등에서 직접 참조될 수 있음)
        MODULE_STATE_DIR = current_process_root / "ai_states"
        MODULE_ART_DIR = current_process_root / "artifacts"

        # 애플리케이션 실행에 필요한 디렉토리 생성 (Handover 인스턴스 내부에서도 호출 가능)
        MODULE_STATE_DIR.mkdir(exist_ok=True)
        MODULE_ART_DIR.mkdir(exist_ok=True)

        # ArgumentParser 설정
        parser = argparse.ArgumentParser(
            description="AI 기반 프로젝트 인수인계 상태 관리 도구 (v1.1.5)", 
            formatter_class=argparse.RawTextHelpFormatter # 도움말 형식 유지
        )
        
        backend_choices_list = list(available_backends.keys()) if available_backends else []
        default_backend_choice = "none" # AI 기능 없이 실행이 기본
        if "ollama" in backend_choices_list: 
            default_backend_choice = "ollama" # ollama가 있으면 기본으로
        elif backend_choices_list: # 그 외 사용 가능한 백엔드가 있다면 첫번째 것
            default_backend_choice = backend_choices_list[0]
        
        parser.add_argument(
            "--backend", 
            default=os.getenv("AI_BACKEND", default_backend_choice),
            choices=backend_choices_list + ["none"], # "none"을 명시적 선택지로 추가
            help=f"사용할 AI 백엔드 선택 (기본값: 환경변수 AI_BACKEND 또는 '{default_backend_choice}'). "
                 f"사용 가능: {', '.join(backend_choices_list) or '없음'}. 'none'으로 AI 기능 비활성화."
        )
        
        # Subparsers 설정
        subparsers = parser.add_subparsers(dest="command", help="실행할 작업", required=True) # 명령어 필수
        
        command_configurations = [
            ("save", "현재 작업 상태를 요약하여 Git에 저장", True, True), # 이름, 도움말, Git필요, AI필요
            ("load", "과거에 저장된 작업 상태를 불러오기", True, True),
            ("diff", "현재 작업 디렉토리의 변경 사항 미리보기 (Git)", True, False),
            ("verify", "저장된 상태의 아티팩트 체크섬 정보 표시", True, False)
        ]
        
        for cmd_name, cmd_help, needs_git_flag, needs_ai_flag in command_configurations:
            cmd_parser = subparsers.add_parser(cmd_name, help=f"{cmd_help}{' (Git 필요)' if needs_git_flag else ''}{' (AI 필요)' if needs_ai_flag else ''}")
            if cmd_name == "load": 
                cmd_parser.add_argument("-l", "--latest", action="store_true", help="가장 최근에 저장된 상태를 자동으로 로드")
            if cmd_name == "diff": 
                cmd_parser.add_argument("target", nargs="?", default="HEAD", help="비교 대상 Commit 또는 Branch (기본값: HEAD)")
            if cmd_name == "verify": 
                cmd_parser.add_argument("commit", help="체크섬 정보를 확인할 상태의 커밋 해시")
        
        args = parser.parse_args() # 명령줄 인자 파싱

        # 선택된 명령어에 따른 전제 조건 검사
        selected_command_config = next(c for c in command_configurations if c[0] == args.command)
        _, _, cmd_needs_git, cmd_needs_ai = selected_command_config

        if cmd_needs_git and not is_actually_in_git_repo:
            UI.error(f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 감지된 루트: {current_process_root})")
            sys.exit(1)
        
        if cmd_needs_ai:
            if not available_backends and args.backend != "none":
                 UI.error(f"'{args.command}' 명령 실행 불가: 사용 가능한 AI 백엔드가 없습니다. 'backends' 폴더를 확인하세요.")
                 sys.exit(1)
            if args.backend == "none":
                 UI.error(f"'{args.command}' 명령 실행 불가: AI 기능이 필요한 작업이지만 AI 백엔드가 'none'으로 설정되었습니다. '--backend' 옵션으로 AI 백엔드를 지정하세요.")
                 sys.exit(1)

        # 스크립트 정보 출력
        print(f"[bold underline]Handover 스크립트 v1.1.5[/]")
        if is_actually_in_git_repo: 
            print(f"[dim]프로젝트 루트 (Git): {current_process_root}[/]")
        else: 
            print(f"[dim]현재 작업 폴더 (Git 저장소 아님): {current_process_root}[/]")

        # Handover 로직 핸들러 실행
        try:
            # Handover 인스턴스에 결정된 루트 경로 전달
            handler_instance = Handover(backend_choice=args.backend, current_app_root=current_process_root)
            
            if args.command == "save": handler_instance.save()
            elif args.command == "load": handler_instance.load(latest=args.latest)
            elif args.command == "diff": handler_instance.diff(target=args.target)
            elif args.command == "verify": handler_instance.verify_checksums(commit_hash=args.commit)
        
        except Exception as e_handler_execution: # 핸들러 실행 중 발생하는 모든 예외 처리
            UI.error("핸들러 실행 중 예기치 않은 오류 발생", traceback.format_exc())
            sys.exit(1)

    if __name__ == "__main__":
        if sys.version_info < (3, 8): # 파이썬 버전 호환성 확인
             print("[bold red]오류: 이 스크립트는 Python 3.8 이상 버전이 필요합니다.[/]")
             sys.exit(1)
        main_application_entry() # 메인 함수 호출



