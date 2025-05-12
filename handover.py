#!/usr/bin/env python3
# handover.py – 인수인계 v1.1.5 (Serializer._calculate_sha256 구문 최종 수정 및 경로 처리 개선)

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

INITIAL_ROOT_PATH = pathlib.Path('.').resolve()
BACKEND_DIR = pathlib.Path(__file__).resolve().parent / "backends"
COMMIT_TAG = "state("

try:
    base_spec = importlib.util.spec_from_file_location("backends.base", BACKEND_DIR / "base.py")
    if base_spec is None or base_spec.loader is None:
        raise ImportError(f"backends.base 모듈 스펙을 찾을 수 없습니다. 경로: {BACKEND_DIR / 'base.py'}")
    backends_base_module = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(backends_base_module)
    AIBaseBackend = backends_base_module.AIBaseBackend
except ImportError as e:
    print(f"[bold red]오류: backends.base 모듈 임포트 실패: {e}[/]")
    sys.exit(1)
except AttributeError:
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
            spec = importlib.util.spec_from_file_location(f"backends.{module_name_stem}", f_py)
            if spec is None or spec.loader is None: continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, obj in module.__dict__.items():
                if (isinstance(obj, type) and
                        issubclass(obj, AIBaseBackend) and
                        obj is not AIBaseBackend):
                    backend_name_from_class = obj.get_name()
                    if backend_name_from_class != "base":
                        available_backends[backend_name_from_class] = obj
        except Exception as e:
            print(f"[yellow]경고: 백엔드 파일 {f_py.name} 처리 중 예외: {e}[/]")
else:
    print(f"[yellow]경고: 백엔드 디렉토리 '{BACKEND_DIR}'를 찾을 수 없거나 디렉토리가 아닙니다.[/]")

class AIProvider:
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        if not available_backends and backend_name != "none":
             raise RuntimeError("AIProvider 초기화 실패: 사용 가능한 AI 백엔드가 없습니다.")
        if backend_name != "none" and backend_name not in available_backends:
            raise ValueError(f"알 수 없는 백엔드: '{backend_name}'. 사용 가능: {list(available_backends.keys()) + ['none']}")
        if backend_name == "none":
            self.backend = None
            print(f"[dim]AI 백엔드: [bold yellow]none (비활성화됨)[/][/dim]")
            return
        BackendClass = available_backends[backend_name]
        try:
            self.backend: Optional[AIBaseBackend] = BackendClass(config)
            print(f"[dim]AI 백엔드 사용: [bold cyan]{backend_name}[/][/dim]")
        except Exception as e:
            print(f"[bold red]오류: 백엔드 '{backend_name}' 초기화 실패: {e}[/]")
            if hasattr(BackendClass, 'get_config_description'):
                print(f"[yellow]필요 설정:\n{BackendClass.get_config_description()}[/]")
            raise e
    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        if not self.backend: raise RuntimeError("AI 백엔드가 'none'으로 설정되어 요약을 생성할 수 없습니다.")
        return self.backend.make_summary(task, ctx, arts)
    def verify_summary(self, md: str) -> Tuple[bool, str]:
        if not self.backend: raise RuntimeError("AI 백엔드가 'none'으로 설정되어 요약을 검증할 수 없습니다.")
        is_ok, msg = self.backend.verify_summary(md)
        if is_ok:
           lines = md.strip().split('\n'); headers = [l.strip() for l in lines if l.startswith('#')]
           req_struct = ["#", "## 목표", "## 진행", "## 결정", "## 결과", "## 다음할일", "## 산출물"]
           if len(headers) != len(req_struct): is_ok, msg = False, f"헤더 개수 불일치 (필수 {len(req_struct)}, 현재 {len(headers)})"
           else:
               for i, r_start in enumerate(req_struct):
                   if not headers[i].startswith(r_start):
                       if i == 0 and headers[i].startswith("# "): continue # 메인 제목은 '# ' 허용
                       is_ok, msg = False, f"헤더 #{i+1} 형식 또는 순서 오류: '{headers[i]}' (예상: '{r_start} 이름')"; break
        return is_ok, msg
    def load_report(self, md: str) -> str:
        if not self.backend: raise RuntimeError("AI 백엔드가 'none'으로 설정되어 보고서를 로드할 수 없습니다.")
        return self.backend.load_report(md)

class GitRepo:
    def __init__(self, repo_path: pathlib.Path):
        try: self.repo = Repo(repo_path)
        except InvalidGitRepositoryError: raise
    def _safe(self, git_func, *args, **kwargs):
        try: return git_func(*args, **kwargs)
        except GitCommandError as e: stderr = e.stderr.strip(); raise RuntimeError(f"Git 명령어 실패: {e.command}\n오류: {stderr}") from e
    def get_last_state_commit(self) -> Optional[Commit]:
        try:
            for c in self.repo.iter_commits(max_count=200, first_parent=True):
                if c.message.startswith(COMMIT_TAG): return c
        except Exception: pass; return None
    def get_commit_messages_since(self, commit_hash: Optional[str]) -> str:
        if not self.repo: return "Git 저장소 초기화 안됨."
        if not commit_hash:
            try: commits = list(self.repo.iter_commits(max_count=10, no_merges=True)); log = "\n".join(f"- {c.hexsha[:7]}: {c.summary}" for c in reversed(commits)); return f"최근 커밋 {len(commits)}개:\n{log}" if log else "최근 커밋 없음"
            except Exception as e: return f"최근 커밋 로그 조회 실패: {e}"
        try: self.repo.commit(commit_hash); log_cmd = f"{commit_hash}..HEAD"; commit_log = self.repo.git.log(log_cmd, '--pretty=format:- %h: %s', '--abbrev-commit', '--no-merges'); return f"'{commit_hash[:8]}' 이후 커밋:\n{commit_log}" if commit_log else f"'{commit_hash[:8]}' 이후 커밋 없음"
        except Exception as e: return f"커밋 로그 조회 중 오류 ({commit_hash}): {e}"
    def get_current_branch(self) -> Optional[str]:
        if not self.repo: return "Git 저장소 없음"
        try: return self.repo.active_branch.name
        except TypeError:
            try: return f"DETACHED_HEAD@{self.repo.head.commit.hexsha[:7]}"
            except Exception: return "DETACHED_HEAD"
        except Exception: return None
    def get_diff(self, target: str = "HEAD", color: bool = True) -> str:
        if not self.repo: return "Git 저장소 초기화 안됨."
        try:
            self.repo.commit(target)
            color_opt = '--color=always' if color else '--color=never'
            staged_diff = self.repo.git.diff('--staged', target, color_opt)
            working_tree_vs_target_diff = self.repo.git.diff(target, color_opt)
            diff_output = ""
            has_staged = bool(staged_diff.strip()); has_wt_vs_target = bool(working_tree_vs_target_diff.strip())
            if has_staged: diff_output += f"--- Staged Changes (vs {target}) ---\n{staged_diff}\n\n"
            if has_wt_vs_target and working_tree_vs_target_diff != staged_diff: diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"
            elif not has_staged and has_wt_vs_target: diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"
            return diff_output.strip() if diff_output.strip() else f"'{target}'과(와) 변경 사항 없음 (Git 추적 파일 기준)"
        except Exception as e: return f"Diff 생성 오류: {e}"
    def save(self, state_paths: List[pathlib.Path], task: str, snapshot_dir: Optional[pathlib.Path]) -> str:
        if not self.repo: raise RuntimeError("Git 저장소 없음.")
        paths_to_add_str = [str(p.resolve()) for p in state_paths]
        if snapshot_dir and snapshot_dir.exists() and any(snapshot_dir.iterdir()): paths_to_add_str.append(str(snapshot_dir.resolve()))
        if not state_paths: raise ValueError("저장할 상태 파일 없음.")
        self._safe(self.repo.git.add, *paths_to_add_str)
        commit_msg = f"{COMMIT_TAG}{task})"
        try: self._safe(self.repo.index.commit, commit_msg)
        except RuntimeError as e:
            if "nothing to commit" in str(e).lower() or "no changes added to commit" in str(e).lower():
                return self.repo.head.commit.hexsha[:8] + " (변경 없음)"
            raise e
        if self.repo.remotes:
            try:
                current_branch_name = self.get_current_branch()
                if current_branch_name and not current_branch_name.startswith("DETACHED_HEAD"):
                    self._safe(self.repo.git.push, 'origin', current_branch_name)
                    print("[green]원격 푸시 완료.[/]")
                else:
                    print(f"[yellow]경고: 현재 브랜치({current_branch_name})를 특정할 수 없어 푸시를 건너뜁니다.[/]")
            except RuntimeError as e:
                print(f"[yellow]경고: 원격 저장소 푸시 실패. 로컬에는 커밋되었습니다. ({e})[/]")
        else:
            print("[yellow]경고: 설정된 원격 저장소가 없어 푸시를 건너뜁니다.[/]")
        return self.repo.head.commit.hexsha[:8]
    def list_states(self, current_app_state_dir: pathlib.Path) -> List[Dict]:
        if not self.repo: return []
        items = [];
        search_path_str = str(current_app_state_dir.relative_to(self.repo.working_dir)) if current_app_state_dir.is_absolute() and self.repo.working_dir in current_app_state_dir.parents else current_app_state_dir.name
        try: commits = list(self.repo.iter_commits(max_count=100, first_parent=True, paths=search_path_str))
        except Exception: commits = list(self.repo.iter_commits(max_count=100, first_parent=True))
        for c in commits:
            if not c.message.startswith(COMMIT_TAG): continue
            headline = ""; meta_blob = None
            try:
                commit_state_dir_name = current_app_state_dir.name
                # 경로 구분자를 시스템에 맞게 처리 (as_posix는 슬래시로 고정)
                commit_state_dir_path_prefix = pathlib.Path(self.repo.working_dir).relative_to(self.repo.working_dir) / commit_state_dir_name
                commit_state_dir_path_prefix_str = commit_state_dir_path_prefix.as_posix()

                # tree[key] 접근 대신 traverse로 경로 기반 검색
                for item in c.tree.traverse():
                     # item.path 가 state 디렉토리 하위에 있고 .meta.json 으로 끝나는지 확인
                     if isinstance(item, Blob) and item.path.startswith(commit_state_dir_path_prefix_str) and item.path.endswith(".meta.json"):
                         meta_blob = item; break

                if meta_blob: metadata = json.loads(meta_blob.data_stream.read().decode('utf-8')); headline = metadata.get("headline", "")
            except Exception as e: headline = f"[메타데이터 오류: {e}]" # 디버깅 위해 오류 메시지 포함
            items.append({"hash": c.hexsha[:8], "task": c.message[len(COMMIT_TAG):-1].strip(), "time": datetime.datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d %H:%M"), "head": headline or "-"})
        return list(reversed(items))
    def load_state(self, commit_hash: str, current_app_state_dir: pathlib.Path) -> str:
        if not self.repo: raise RuntimeError("Git 저장소 없음.")
        try: commit_obj = self.repo.commit(self.repo.git.rev_parse(commit_hash))
        except Exception as e: raise RuntimeError(f"커밋 '{commit_hash}' 접근 오류: {e}") from e
        try:
            commit_state_dir_name = current_app_state_dir.name
            # list_states와 유사하게 경로 기반 탐색
            commit_state_dir_path_prefix = pathlib.Path(self.repo.working_dir).relative_to(self.repo.working_dir) / commit_state_dir_name
            commit_state_dir_path_prefix_str = commit_state_dir_path_prefix.as_posix()

            for item in commit_obj.tree.traverse():
                 if isinstance(item, Blob) and item.path.startswith(commit_state_dir_path_prefix_str) and item.path.endswith(".md"):
                     return item.data_stream.read().decode('utf-8')

            raise RuntimeError(f"커밋 '{commit_hash}' 내 '{commit_state_dir_path_prefix_str}' 경로에 .md 파일 없음.")
        except Exception as e: raise RuntimeError(f"'{commit_hash}' 상태 로드 중 오류: {e}")

class Serializer:
    @staticmethod
    def _calculate_sha256(fp: pathlib.Path) -> Optional[str]:
        h = hashlib.sha256()
        try:
            with open(fp, "rb") as f:
                while True:
                    b = f.read(4096)  # 한 줄에 하나의 명령
                    if not b:
                        break         # if 문 안에 break 만 오도록 하고, 다음 줄로 분리
                    h.update(b)       # h.update(b)도 별도의 줄로 분리하고 들여쓰기 맞춤
            return h.hexdigest()
        except IOError:
            # print(f"[yellow]경고: 파일 해시 계산 중 IO오류 ({fp.name})[/]") # 상세 오류는 선택
            return None
        except Exception as e:
            # print(f"[yellow]경고: 파일 해시 계산 중 예외 ({fp.name}): {e}[/]") # 상세 오류는 선택
            return None

    @staticmethod
    def _generate_html(md: str, title: str) -> str:
       css = """<style>body{font-family:sans-serif;line-height:1.6;padding:20px;max-width:800px;margin:auto;color:#333}h1,h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.5em;margin-bottom:1em}h1{font-size:2em}h2{font-size:1.5em}ul,ol{padding-left:2em}li{margin-bottom:.5em}code{background-color:#f0f0f0;padding:.2em .4em;border-radius:3px;font-family:monospace;font-size:.9em}pre{background-color:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}pre code{background-color:transparent;padding:0;border-radius:0}blockquote{border-left:4px solid #ccc;padding-left:1em;color:#666;margin-left:0}table{border-collapse:collapse;width:100%;margin-bottom:1em}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}</style>"""
       body = markdown2.markdown(md, extras=["metadata","fenced-code-blocks","tables","strike","task_list","code-friendly","markdown-in-html"])
       title_meta = title;
       if hasattr(body,"metadata") and body.metadata.get("title"): title_meta = body.metadata["title"]
       return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{textwrap.shorten(title_meta, width=50, placeholder="...")}</title>{css}</head><body>{body}</body></html>"""

    @staticmethod
    def save_state(md: str, task: str, current_app_state_dir: pathlib.Path, current_app_art_dir: pathlib.Path, current_app_root: pathlib.Path) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S"); safe_task = "".join(c for c in task if c.isalnum() or c in (' ','_','-')).strip().replace(' ','_');
        if not safe_task: safe_task="untitled_task";
        base_fn = f"{ts}_{safe_task}"
        current_app_state_dir.mkdir(exist_ok=True); current_app_art_dir.mkdir(exist_ok=True)
        state_f = current_app_state_dir / f"{base_fn}.md"; html_f = current_app_state_dir / f"{base_fn}.html"; meta_f = current_app_state_dir / f"{base_fn}.meta.json"
        try: state_f.write_text(md, encoding="utf-8")
        except IOError as e: raise RuntimeError(f"MD 파일 저장 실패 ({state_f}): {e}") from e
        html_ok = False
        try: html_f.write_text(Serializer._generate_html(md,task),encoding="utf-8"); html_ok=True
        except Exception as e: print(f"[yellow]경고: HTML 생성 실패 ({html_f.name}): {e}[/]")
        snap_dir = None; checksums = {}; arts = [f for f in current_app_art_dir.iterdir() if f.is_file()] if current_app_art_dir.exists() else []
        if arts:
            snapshot_sub_dir_name = f"{base_fn}_artifacts"
            snap_dir = current_app_art_dir / snapshot_sub_dir_name
            snap_dir.mkdir(parents=True,exist_ok=True)
            print(f"[dim]아티팩트 스냅샷 ({len(arts)}개) -> '{snap_dir.relative_to(current_app_root)}'[/]")
            for f_art in arts:
                try:
                    target_path = snap_dir / f_art.name
                    shutil.copy2(f_art, target_path)
                    cs = Serializer._calculate_sha256(target_path)
                    if cs: checksums[f_art.name] = cs
                except Exception as copy_e: print(f"[yellow]경고: 아티팩트 파일 '{f_art.name}' 복사/해시 실패: {copy_e}[/]")
        else: print("[dim]저장할 아티팩트 파일이 없습니다.[/]")

        # --- 오류 발생 지점 수정 ---
        # 원본: for ln in md.splitlines(): ln_s = ln.strip(); if ln_s.startswith("#"): headline=ln_s.lstrip('# ').strip(); break
        headline = task # 기본값 설정
        for ln in md.splitlines():
            ln_s = ln.strip()
            if ln_s.startswith("#"): # 첫 번째 '#'으로 시작하는 줄을 헤드라인으로 간주
                headline = ln_s.lstrip('# ').strip() # '#'과 앞뒤 공백 제거
                break # 헤드라인을 찾으면 루프 종료
        # --- 수정 끝 ---

        meta = {"task":task,"ts":ts,"headline":headline,"artifact_checksums":checksums}
        try: meta_f.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        except IOError as e: raise RuntimeError(f"메타데이터 파일 저장 실패 ({meta_f}): {e}") from e
        paths_to_commit = [state_f, meta_f];
        if html_ok and html_f.exists(): paths_to_commit.append(html_f)
        valid_snap_dir = snap_dir if (snap_dir and snap_dir.exists() and any(snap_dir.iterdir())) else None
        return paths_to_commit, valid_snap_dir
    @staticmethod
    def to_prompt(md: str, commit: str) -> str: return f"### 이전 상태 (Commit: {commit}) ###\n\n{md}\n\n### 상태 정보 끝 ###"

class UI:
    console = Console()
    @staticmethod
    def task_name(default:str="작업 요약") -> str: return Prompt.ask("[bold cyan]작업 이름[/]",default=default)
    @staticmethod
    def multiline(label: str, default: str = "") -> str:
        UI.console.print(f"\n[bold cyan]{label}[/]"); UI.console.print("[dim](입력 완료: 빈 줄에서 Enter 두 번)[/]")
        lines = []
        if default:
            default_preview = textwrap.shorten(default, width=100, placeholder="...") if len(default.splitlines()) > 5 else default
            print(Panel(default_preview,title="[dim]자동 제안 (편집 가능, 전체 내용은 아래 입력)[/]",border_style="dim",expand=False))
        blank_count = 0
        while True:
            try: line = input()
            except EOFError: break
            if line=="": blank_count+=1;
            if blank_count>=2: break
            else: blank_count=0; lines.append(line)
        final = "\n".join(lines).strip();
        if not final and default: print("[dim]입력 없음, 제안 내용 사용.[/]"); return default
        return final
    @staticmethod
    def notify(msg:str,style:str="green"): UI.console.print(f"\n[bold {style}]✔ {msg}[/]")
    @staticmethod
    def error(msg:str,details:Optional[str]=None):
        UI.console.print(f"\n[bold red]❌ 오류: {msg}[/]")
        if details: UI.console.print(Panel(details,title="[dim]상세 정보 (Traceback)[/]",border_style="dim red",expand=False, height=15 if len(details.splitlines()) > 15 else None))
    @staticmethod
    def pick_state(states:List[Dict])->Optional[str]:
        if not states: print("[yellow]저장된 상태가 없습니다.[/]"); return None
        tb = Table(title="[bold]저장된 인수인계 상태 목록[/]",box=box.ROUNDED,show_lines=True, expand=False)
        tb.add_column("#",style="dim",justify="right", width=3); tb.add_column("커밋", style="cyan", no_wrap=True, width=10); tb.add_column("작업", style="magenta", min_width=20, overflow="fold"); tb.add_column("시각", style="green", no_wrap=True, width=18); tb.add_column("헤드라인", style="yellow", overflow="fold", min_width=30)
        for i,s in enumerate(states):tb.add_row(str(i),s["hash"],s["task"],s["time"],s["head"])
        UI.console.print(tb); choices=[str(i) for i in range(len(states))]
        sel=Prompt.ask("[bold cyan]로드할 상태 번호 (취소하려면 Enter)[/]",choices=choices+[""],default="",show_choices=False)
        if sel.isdigit() and 0 <= int(sel) < len(states):
            s_hash=states[int(sel)]["hash"]; print(f"[info]선택된 커밋: {s_hash} (작업: {states[int(sel)]['task']})[/]"); return s_hash
        print("[info]상태 로드를 취소했습니다.[/]"); return None
    @staticmethod
    def panel(txt:str,title:str,border_style:str="blue"): UI.console.print(Panel(txt,title=f"[bold]{title}[/]",border_style=border_style,expand=False,padding=(1,2)))
    @staticmethod
    def diff_panel(txt:str,target:str):
        if not txt.strip() or "변경 사항 없음" in txt or txt.startswith("Diff 생성 오류"): print(f"[dim]{txt}[/]"); return
        syntax_obj = Syntax(txt,"diff",theme="default",line_numbers=False,word_wrap=False)
        UI.console.print(Panel(syntax_obj,title=f"[bold]Diff (vs {target})[/]",border_style="yellow",expand=True))

class Handover:
    def __init__(self, backend_choice: str, current_app_root: pathlib.Path):
        self.ui = UI(); self.app_root = current_app_root
        self.state_dir = self.app_root / "ai_states"
        self.art_dir = self.app_root / "artifacts"
        try: self.git = GitRepo(self.app_root)
        except InvalidGitRepositoryError: self.git = None # Git 저장소가 아니어도 일단 진행 가능하도록 None 할당
        except Exception as e: self.ui.error(f"GitRepo 초기화 실패: {e}", traceback.format_exc()); sys.exit(1)
        if backend_choice != "none" and available_backends:
            try: self.ai = AIProvider(backend_name=backend_choice, config={})
            except Exception as e: self.ui.error(f"AI 백엔드 ('{backend_choice}') 초기화 실패.", traceback.format_exc()); sys.exit(1)
        elif backend_choice != "none" and not available_backends: self.ui.error(f"선택된 AI 백엔드 '{backend_choice}'를 위한 모듈을 찾을 수 없거나 로드 중 오류 발생."); sys.exit(1)
        else: self.ai = None; self.ui.console.print("[yellow]경고: AI 백엔드가 'none'으로 설정되어 AI 기능이 비활성화됩니다.[/]")

    def _ensure_prereqs(self,cmd:str,needs_git:bool,needs_ai:bool):
        if needs_git and not self.git: self.ui.error(f"'{cmd}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 루트: {self.app_root})"); sys.exit(1)
        if needs_ai and not self.ai: self.ui.error(f"'{cmd}' 명령은 AI 백엔드가 설정되어야 합니다. ('--backend' 옵션을 확인하거나 'none'이 아닌지 확인하세요.)"); sys.exit(1)

    def save(self):
        self._ensure_prereqs("save", True, True) # save는 Git과 AI 모두 필요
        try:
            default_task_name = "작업 요약" # 기본값 안전하게 설정
            if self.git: # self.git 이 None이 아닐 경우에만 접근
                try:
                     # 현재 브랜치 또는 마지막 커밋 요약 가져오기 시도
                     current_branch = self.git.get_current_branch()
                     if current_branch and not current_branch.startswith("DETACHED_HEAD"):
                         default_task_name = current_branch
                     elif self.git.repo and self.git.repo.head.is_valid():
                         default_task_name = self.git.repo.head.commit.summary
                except Exception as e_git_info:
                     print(f"[yellow]경고: 기본 작업 이름 설정 중 오류 ({e_git_info})[/]")

            task_name_input = self.ui.task_name(default=default_task_name)
            last_saved_commit = self.git.get_last_state_commit() if self.git else None
            default_context_summary = self.git.get_commit_messages_since(last_saved_commit.hexsha if last_saved_commit else None) if self.git else "최근 변경사항 없음"
            context_summary_input = self.ui.multiline("작업 내용 요약 (AI가 생성한 요약을 붙여넣거나 직접 작성)", default=default_context_summary)
            if not context_summary_input.strip(): self.ui.error("작업 내용 요약이 비어있어 저장을 취소합니다."); return
            self.state_dir.mkdir(exist_ok=True); self.art_dir.mkdir(exist_ok=True)
            current_artifacts = [f.name for f in self.art_dir.iterdir() if f.is_file()] if self.art_dir.exists() else []
            if self.art_dir.exists(): self.ui.console.print(f"[dim]현재 아티팩트 ({self.art_dir.relative_to(self.app_root)}): {', '.join(current_artifacts) or '없음'}[/]")
            else: self.ui.console.print(f"[dim]아티팩트 폴더({self.art_dir.relative_to(self.app_root)})가 존재하지 않습니다.[/]")
            self.ui.console.print("\n[bold yellow]AI가 인수인계 문서를 생성 중입니다...[/]")
            generated_markdown = self.ai.make_summary(task_name_input, context_summary_input, current_artifacts)
            self.ui.panel(generated_markdown, "AI 생성 요약본 (검증 전)")
            self.ui.console.print("[bold yellow]생성된 요약본을 AI가 검증 중입니다...[/]")
            is_valid_summary, validation_message = self.ai.verify_summary(generated_markdown)
            if not is_valid_summary: raise RuntimeError(f"AI가 생성한 인수인계 문서 검증 실패:\n{validation_message}")
            self.ui.notify("AI 검증 통과!", style="green")
            saved_state_files, artifact_snapshot_dir = Serializer.save_state(generated_markdown, task_name_input, self.state_dir, self.art_dir, self.app_root)
            commit_short_hash = self.git.save(saved_state_files, task_name_input, artifact_snapshot_dir) if self.git else "N/A (No Git)"
            self.ui.notify(f"인수인계 상태 저장 완료! (Commit: {commit_short_hash})", style="bold green")
            generated_html_file = next((f for f in saved_state_files if f.name.endswith(".html")), None)
            if generated_html_file and generated_html_file.exists():
                 self.ui.console.print(f"[dim]HTML 프리뷰 생성됨: {generated_html_file.relative_to(self.app_root)}[/]")
        except Exception as e: self.ui.error(f"Save 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def load(self, latest: bool = False):
        self._ensure_prereqs("load", True, True) # load는 Git과 AI 모두 필요
        try:
            saved_states_list = self.git.list_states(self.state_dir) if self.git else []
            if not saved_states_list: self.ui.error("저장된 인수인계 상태가 없습니다."); return
            selected_commit_hash: Optional[str]
            if latest:
                selected_commit_hash = saved_states_list[-1]["hash"]
                print(f"[info]가장 최근 상태 로드 중: {selected_commit_hash} (작업명: {saved_states_list[-1]['task']})[/]")
            else: selected_commit_hash = self.ui.pick_state(saved_states_list)
            if not selected_commit_hash: return
            self.ui.console.print(f"[bold yellow]{selected_commit_hash} 커밋에서 상태 정보를 로드 중입니다...[/]")
            markdown_content = self.git.load_state(selected_commit_hash, self.state_dir) if self.git else "Error: Git not available"
            prompt_formatted_content = Serializer.to_prompt(markdown_content, selected_commit_hash)
            self.ui.panel(prompt_formatted_content, f"로드된 상태 (Commit: {selected_commit_hash})", border_style="cyan")
            self.ui.console.print("[bold yellow]AI가 로드된 상태를 분석하고 이해도를 보고합니다...[/]")
            ai_report = self.ai.load_report(markdown_content) if self.ai else "AI is not available."
            self.ui.panel(ai_report, "AI 이해도 보고서", border_style="magenta")
        except Exception as e: self.ui.error(f"Load 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def diff(self, target: str = "HEAD"):
        self._ensure_prereqs("diff", True, False) # diff는 Git 필요, AI 불필요
        try:
            self.ui.console.print(f"[bold yellow]'{target}' 대비 현재 변경 사항을 확인 중입니다... (Git 추적 파일 기준)[/]")
            diff_output_text = self.git.get_diff(target, color=True) if self.git else "Error: Git not available."
            self.ui.diff_panel(diff_output_text, target)
        except Exception as e: self.ui.error(f"Diff 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def verify_checksums(self, commit_hash: str):
        self._ensure_prereqs("verify", True, False) # verify는 Git 필요, AI 불필요
        self.ui.console.print(f"[dim]커밋 {commit_hash}의 저장된 아티팩트 체크섬 정보를 표시합니다. (실제 파일 비교 검증은 아직 구현되지 않았습니다.)[/]")
        if not self.git: self.ui.error("Git 저장소가 없어 체크섬 정보를 확인할 수 없습니다."); return
        try:
            commit_obj = self.git.repo.commit(self.git.repo.git.rev_parse(commit_hash))
            meta_blob: Optional[Blob] = None
            # GitRepo.list_states와 동일한 경로 탐색 로직 사용
            commit_state_dir_path_prefix = pathlib.Path(self.git.repo.working_dir).relative_to(self.git.repo.working_dir) / self.state_dir.name
            commit_state_dir_path_prefix_str = commit_state_dir_path_prefix.as_posix()

            for item in commit_obj.tree.traverse():
                if isinstance(item, Blob) and item.path.startswith(commit_state_dir_path_prefix_str) and item.path.endswith(".meta.json"):
                    meta_blob = item; break

            if not meta_blob: self.ui.error(f"커밋 {commit_hash}에서 메타데이터 파일(.meta.json)을 찾을 수 없습니다. (탐색 경로: {commit_state_dir_path_prefix_str})"); return

            metadata_content = json.loads(meta_blob.data_stream.read().decode('utf-8'))
            artifact_checksums_data = metadata_content.get("artifact_checksums", {})
            if artifact_checksums_data:
                checksums_pretty_str = json.dumps(artifact_checksums_data, indent=2, ensure_ascii=False)
                self.ui.panel(checksums_pretty_str, f"저장된 아티팩트 체크섬 (Commit: {commit_hash})", border_style="magenta")
            else: print(f"[dim]커밋 {commit_hash}에 저장된 아티팩트 체크섬 정보가 없습니다.[/]")
        except GitCommandError: self.ui.error(f"유효한 커밋 해시가 아니거나 찾을 수 없습니다: '{commit_hash}'.")
        except Exception as e: self.ui.error(f"체크섬 정보 로드/표시 중 오류 ({commit_hash}): {str(e)}", traceback.format_exc())

def main_cli_entry_point():
    # 전역 변수 제거, 필요한 경우 함수 인자로 전달하거나 지역 변수로 사용
    cli_root_path = pathlib.Path('.').resolve()
    git_repo_root_obj = None
    is_git_repo_at_cli_root = False
    try:
        # search_parent_directories=True 로 상위 폴더까지 검색
        git_repo_root_obj = Repo(cli_root_path, search_parent_directories=True)
        # 실제 Git 저장소 루트 경로 사용
        cli_root_path = pathlib.Path(git_repo_root_obj.working_tree_dir)
        is_git_repo_at_cli_root = True
    except InvalidGitRepositoryError:
        # Git 저장소가 아니어도 cli_root_path는 현재 작업 디렉토리로 유지됨
        pass
    except Exception as e:
        # 다른 예외 발생 시 (예: git 실행파일 못찾음) 경고 출력
        print(f"[yellow]경고: Git 저장소 확인 중 오류 발생: {e}[/]")


    # Handover 클래스에서 사용할 경로들은 cli_root_path 기준으로 생성
    app_state_dir = cli_root_path / "ai_states"
    app_art_dir = cli_root_path / "artifacts"
    app_state_dir.mkdir(exist_ok=True)
    app_art_dir.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="AI 기반 프로젝트 인수인계 상태 관리 도구 (v1.1.5)", formatter_class=argparse.RawTextHelpFormatter)
    backend_choices_list = list(available_backends.keys()) if available_backends else []
    default_be = "none"
    if "ollama" in backend_choices_list: default_be = "ollama"
    elif backend_choices_list: default_be = backend_choices_list[0]
    parser.add_argument("--backend", default=os.getenv("AI_BACKEND", default_be), choices=backend_choices_list + ["none"], help=f"AI 백엔드 (기본값: 환경변수 AI_BACKEND 또는 '{default_be}'). 사용가능: {', '.join(backend_choices_list) or '없음'}. 'none'으로 AI 비활성화.")
    subparsers = parser.add_subparsers(dest="command", help="실행할 작업", required=True)
    cmd_configs = [("save", "현재 작업 상태를 요약/저장", True, True), ("load", "과거 저장된 상태 불러오기", True, True), ("diff", "현재 변경 사항 미리보기 (Git)", True, False), ("verify", "저장된 상태 아티팩트 체크섬 표시", True, False)]
    for name, help_txt, git_req, ai_req in cmd_configs:
        p = subparsers.add_parser(name, help=f"{help_txt}{' (Git 필요)' if git_req else ''}{' (AI 필요)' if ai_req else ''}")
        if name == "load": p.add_argument("-l", "--latest", action="store_true", help="가장 최근 상태 자동 로드")
        if name == "diff": p.add_argument("target", nargs="?", default="HEAD", help="비교 대상 Commit/Branch (기본값: HEAD)")
        if name == "verify": p.add_argument("commit", help="체크섬 정보를 확인할 상태 커밋 해시")
    args = parser.parse_args()

    chosen_cmd_config = next(c for c in cmd_configs if c[0] == args.command)
    # Git 필요 여부 체크 시 is_git_repo_at_cli_root 변수 사용
    if chosen_cmd_config[2] and not is_git_repo_at_cli_root: UI.error(f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 검색된 루트: {cli_root_path})"); sys.exit(1)
    if chosen_cmd_config[3] and (not available_backends or args.backend == "none"):
        msg = "사용 가능한 AI 백엔드가 없습니다. 'backends' 폴더 확인." if not available_backends else f"AI 백엔드가 '{args.backend}'(으)로 설정됨."
        UI.error(f"'{args.command}' 명령 실행 불가: {msg} AI 기능이 필요합니다."); sys.exit(1)

    print(f"[bold underline]Handover 스크립트 v1.1.5[/]")
    if is_git_repo_at_cli_root: print(f"[dim]프로젝트 루트 (Git): {cli_root_path}[/]")
    else: print(f"[dim]현재 작업 폴더 (Git 저장소 아님): {cli_root_path}[/]")

    try:
        # Handover 초기화 시 실제 Git 루트 또는 현재 작업 폴더 경로 전달
        handler = Handover(backend_choice=args.backend, current_app_root=cli_root_path)
        if args.command == "save": handler.save()
        elif args.command == "load": handler.load(latest=args.latest)
        elif args.command == "diff": handler.diff(target=args.target)
        elif args.command == "verify": handler.verify_checksums(commit_hash=args.commit)
    except Exception as e_handler: UI.error("핸들러 실행 중 예기치 않은 오류", traceback.format_exc()); sys.exit(1)

if __name__ == "__main__":
    if sys.version_info < (3, 8): print("[bold red]오류: Python 3.8 이상 필요.[/]"); sys.exit(1)
    main_cli_entry_point()