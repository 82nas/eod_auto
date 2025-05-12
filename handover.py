#!/usr/bin/env python3
# handover.py – 인수인계 v1.1.5 (Serializer.save_state 구문 최종 수정)

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
# 이 전역 변수들은 main_cli_entry_point에서 실제 Git 루트 기준으로 업데이트됩니다.
# Serializer는 main_cli_entry_point에서 설정된 이 전역 변수를 직접 참조하거나,
# Handover 인스턴스를 통해 경로를 전달받아 사용합니다.
# 여기서는 Handover 인스턴스를 통해 경로를 전달하는 방식으로 수정합니다.
# STATE_DIR_MODULE_LEVEL = INITIAL_ROOT_PATH / "ai_states"
# ART_DIR_MODULE_LEVEL = INITIAL_ROOT_PATH / "artifacts"
BACKEND_DIR = pathlib.Path(__file__).resolve().parent / "backends"
COMMIT_TAG = "state("

try:
    spec = importlib.util.spec_from_file_location("backends.base", BACKEND_DIR / "base.py")
    if spec is None or spec.loader is None:
        raise ImportError("backends.base 모듈 스펙을 찾을 수 없습니다.")
    backends_base_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backends_base_module)
    AIBaseBackend = backends_base_module.AIBaseBackend
except ImportError as e:
    print(f"[bold red]오류: backends.base 모듈 임포트 실패: {e}[/]")
    sys.exit(1)
except AttributeError:
    print(f"[bold red]오류: backends.base 모듈에서 AIBaseBackend 클래스 찾기 실패.[/]")
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
            print(f"[yellow]경고: 백엔드 파일 {f_py} 처리 중 예외: {e}[/]")
else:
    print(f"[yellow]경고: 백엔드 디렉토리 '{BACKEND_DIR}' 없음.[/]")

class AIProvider:
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        if not available_backends and backend_name != "none":
             raise RuntimeError("AIProvider 초기화 실패: AI 백엔드 없음.")
        if backend_name != "none" and backend_name not in available_backends:
            raise ValueError(f"알 수 없는 백엔드: {backend_name}. 사용 가능: {list(available_backends.keys()) + ['none']}")
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
        if not self.backend: raise RuntimeError("AI 백엔드 'none', 요약 생성 불가.")
        return self.backend.make_summary(task, ctx, arts)
    def verify_summary(self, md: str) -> Tuple[bool, str]:
        if not self.backend: raise RuntimeError("AI 백엔드 'none', 요약 검증 불가.")
        is_ok, msg = self.backend.verify_summary(md)
        if is_ok:
           lines = md.strip().split('\n'); headers = [l.strip() for l in lines if l.startswith('#')]
           req_struct = ["#", "## 목표", "## 진행", "## 결정", "## 결과", "## 다음할일", "## 산출물"]
           if len(headers) != len(req_struct): is_ok, msg = False, f"헤더 개수 불일치 (필수 {len(req_struct)}, 발견 {len(headers)})"
           else:
               for i, r_start in enumerate(req_struct):
                  if not headers[i].startswith(r_start):
                      if i == 0 and headers[i].startswith("# "): continue
                      is_ok, msg = False, f"헤더 #{i+1} 구조 오류: '{headers[i]}' (예상: '{r_start}' 시작)"; break
        return is_ok, msg
    def load_report(self, md: str) -> str:
        if not self.backend: raise RuntimeError("AI 백엔드 'none', 보고서 로드 불가.")
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
                    print(f"[yellow]경고: 현재 브랜치({current_branch_name})를 특정할 수 없어 푸시를 건너뜁니다.[/]") # Corrected string
            except RuntimeError as e: 
                print(f"[yellow]경고: 원격 저장소 푸시 실패. 로컬에는 커밋되었습니다. ({e})[/]")
        else: 
            print("[yellow]경고: 설정된 원격 저장소가 없어 푸시를 건너뜁니다.[/]") # Corrected string
        return self.repo.head.commit.hexsha[:8]
    def list_states(self, current_state_dir: pathlib.Path) -> List[Dict]:
        if not self.repo: return []
        items = [];
        try: commits = list(self.repo.iter_commits(max_count=100, first_parent=True, paths=str(current_state_dir)))
        except Exception: commits = list(self.repo.iter_commits(max_count=100, first_parent=True))
        for c in commits:
            if not c.message.startswith(COMMIT_TAG): continue
            headline = ""; meta_blob = None
            try:
                commit_state_dir_name = current_state_dir.name # Use name for lookup in commit tree
                if commit_state_dir_name in c.tree:
                    for item in c.tree[commit_state_dir_name].traverse():
                        if isinstance(item, Blob) and item.name.endswith(".meta.json"): meta_blob = item; break
                if not meta_blob: # Fallback search if not in exact named dir (e.g. if dir was renamed)
                     for item in c.tree.traverse():
                          if isinstance(item, Blob) and item.path.startswith(commit_state_dir_name) and item.path.endswith(".meta.json"): meta_blob = item; break
                if meta_blob: metadata = json.loads(meta_blob.data_stream.read().decode('utf-8')); headline = metadata.get("headline", "")
            except Exception: headline = "[메타데이터 오류]"
            items.append({"hash": c.hexsha[:8], "task": c.message[len(COMMIT_TAG):-1].strip(), "time": datetime.datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d %H:%M"), "head": headline or "-"})
        return list(reversed(items))
    def load_state(self, commit_hash: str, current_state_dir: pathlib.Path) -> str:
        if not self.repo: raise RuntimeError("Git 저장소 없음.")
        try: commit_obj = self.repo.commit(self.repo.git.rev_parse(commit_hash))
        except Exception as e: raise RuntimeError(f"커밋 '{commit_hash}' 접근 오류: {e}") from e
        try:
            commit_state_dir_name = current_state_dir.name
            if commit_state_dir_name in commit_obj.tree:
                for item in commit_obj.tree[commit_state_dir_name].traverse():
                    if isinstance(item, Blob) and item.name.endswith(".md"): return item.data_stream.read().decode('utf-8')
            for item in commit_obj.tree.traverse(): # Fallback search
                 if isinstance(item, Blob) and item.path.startswith(commit_state_dir_name) and item.path.endswith(".md"): return item.data_stream.read().decode('utf-8')
            raise RuntimeError(f"커밋 '{commit_hash}' 내 '{commit_state_dir_name}' 폴더에 .md 파일 없음.")
        except KeyError: raise RuntimeError(f"커밋 '{commit_hash}'에 '{commit_state_dir_name}' 폴더 없음.")
        except Exception as e: raise RuntimeError(f"'{commit_hash}' 상태 로드 중 오류: {e}")

class Serializer:
    @staticmethod
    def _calculate_sha256(fp: pathlib.Path) -> Optional[str]:
        h = hashlib.sha256()
        try:
            with open(fp, "rb") as f:
                while True: b = f.read(4096);
                if not b: break; h.update(b)
            return h.hexdigest()
        except IOError: return None
    @staticmethod
    def _generate_html(md: str, title: str) -> str:
         css = """<style>body{font-family:sans-serif;line-height:1.6;padding:20px;max-width:800px;margin:auto;color:#333}h1,h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.5em;margin-bottom:1em}h1{font-size:2em}h2{font-size:1.5em}ul,ol{padding-left:2em}li{margin-bottom:.5em}code{background-color:#f0f0f0;padding:.2em .4em;border-radius:3px;font-family:monospace;font-size:.9em}pre{background-color:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}pre code{background-color:transparent;padding:0;border-radius:0}blockquote{border-left:4px solid #ccc;padding-left:1em;color:#666;margin-left:0}table{border-collapse:collapse;width:100%;margin-bottom:1em}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}</style>"""
         body = markdown2.markdown(md, extras=["metadata","fenced-code-blocks","tables","strike","task_list","code-friendly"])
         title_meta = title;
         if hasattr(body,"metadata") and body.metadata.get("title"): title_meta = body.metadata["title"]
         return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{title_meta}</title>{css}</head><body>{body}</body></html>"""

    @staticmethod
    def save_state(md: str, task: str, state_dir_param: pathlib.Path, art_dir_param: pathlib.Path, current_root: pathlib.Path) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S"); safe_task = "".join(c for c in task if c.isalnum() or c in (' ','_','-')).strip().replace(' ','_');
        if not safe_task: safe_task="untitled_task";
        base_fn = f"{ts}_{safe_task}"

        state_dir_param.mkdir(exist_ok=True); art_dir_param.mkdir(exist_ok=True)

        state_f = state_dir_param / f"{base_fn}.md"; html_f = state_dir_param / f"{base_fn}.html"; meta_f = state_dir_param / f"{base_fn}.meta.json"
        try: state_f.write_text(md, encoding="utf-8")
        except IOError as e: raise RuntimeError(f"MD 파일 저장 실패 ({state_f}): {e}") from e

        html_ok = False
        try: html_f.write_text(Serializer._generate_html(md,task),encoding="utf-8"); html_ok=True
        except Exception as e: print(f"[yellow]경고: HTML 생성 실패 ({html_f.name}): {e}[/]")

        snap_dir = None; checksums = {}; arts = [f for f in art_dir_param.iterdir() if f.is_file()] if art_dir_param.exists() else []
        if arts:
            snap_dir = art_dir_param / f"{base_fn}_artifacts"; snap_dir.mkdir(parents=True,exist_ok=True)
            print(f"[dim]아티팩트 스냅샷 ({len(arts)}개) -> '{snap_dir.relative_to(current_root)}'[/]")
            for f_art in arts:
                # Corrected try-except block for artifact copying and checksum
                try:
                    target_path = snap_dir / f_art.name
                    shutil.copy2(f_art, target_path)
                    cs = Serializer._calculate_sha256(target_path)
                    if cs: 
                        checksums[f_art.name] = cs
                except Exception as copy_e: # Catch any exception during copy or hash
                    print(f"[yellow]경고: 아티팩트 복사/해시 실패 ({f_art.name}): {copy_e}[/]")

        headline = "";
        for ln in md.splitlines(): ln_s = ln.strip(); if ln_s.startswith("#"): headline=ln_s.lstrip('# ').strip(); break
        if not headline: headline = task
        meta = {"task":task,"ts":ts,"headline":headline,"artifact_checksums":checksums}
        try: meta_f.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        except IOError as e: raise RuntimeError(f"메타 파일 저장 실패 ({meta_f}): {e}") from e

        paths = [state_f, meta_f];
        if html_ok and html_f.exists(): paths.append(html_f)
        valid_snap = snap_dir if (snap_dir and snap_dir.exists() and any(snap_dir.iterdir())) else None
        return paths, valid_snap

    @staticmethod
    def to_prompt(md: str, commit: str) -> str: return f"### 이전 상태 (Commit: {commit}) ###\n\n{md}\n\n### 상태 끝 ###"

class UI:
    console = Console()
    @staticmethod
    def task_name(default:str="작업 요약") -> str: return Prompt.ask("[bold cyan]작업 이름[/]",default=default)
    @staticmethod
    def multiline(label: str, default: str = "") -> str:
        UI.console.print(f"\n[bold cyan]{label}[/]"); UI.console.print("[dim](입력 완료: 빈 줄에서 Enter 두 번)[/]")
        lines = []
        if default: print(Panel(default,title="[dim]자동 제안 (편집 가능)[/]",border_style="dim",expand=False))
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
        if details: UI.console.print(Panel(details,title="[dim]상세 정보[/]",border_style="dim red",expand=False))
    @staticmethod
    def pick_state(states:List[Dict])->Optional[str]:
        if not states: print("[yellow]저장된 상태 없음.[/]"); return None
        tb = Table(title="[bold]저장된 인수인계 상태[/]",box=box.ROUNDED,show_lines=True)
        tb.add_column("#",style="dim",justify="right");tb.add_column("커밋");tb.add_column("작업");tb.add_column("시각");tb.add_column("헤드라인",overflow="fold")
        for i,s in enumerate(states):tb.add_row(str(i),s["hash"],s["task"],s["time"],s["head"])
        UI.console.print(tb); choices=[str(i) for i in range(len(states))]
        sel=Prompt.ask("[bold cyan]로드할 상태 번호 (취소: Enter)[/]",choices=choices+[""],default="",show_choices=False)
        if sel.isdigit() and 0<=int(sel)<len(states):
            s_hash=states[int(sel)]["hash"]; print(f"[info]선택: {s_hash} ({states[int(sel)]['task']})[/]"); return s_hash
        print("[info]로드 취소.[/]"); return None
    @staticmethod
    def panel(txt:str,title:str,border_style:str="blue"): UI.console.print(Panel(txt,title=f"[bold]{title}[/]",border_style=border_style,expand=False,padding=(1,2)))
    @staticmethod
    def diff_panel(txt:str,target:str):
        if not txt.strip() or "변경 사항 없음" in txt or "오류" in txt: print(f"[dim]{txt}[/]"); return
        UI.console.print(Panel(Syntax(txt,"diff",theme="default",line_numbers=False,word_wrap=False),title=f"[bold]Diff (vs {target})[/]",border_style="yellow",expand=True))

class Handover:
    def __init__(self, backend_choice: str, current_root: pathlib.Path):
        self.ui = UI(); self.root_dir = current_root
        self.state_dir = self.root_dir / "ai_states"
        self.art_dir = self.root_dir / "artifacts"
        try: self.git = GitRepo(self.root_dir)
        except InvalidGitRepositoryError: self.git = None
        except Exception as e: self.ui.error(f"GitRepo 초기화 실패: {e}", traceback.format_exc()); sys.exit(1)
        if backend_choice != "none" and available_backends:
            try: self.ai = AIProvider(backend_name=backend_choice, config={})
            except Exception as e: self.ui.error(f"AI 백엔드 ({backend_choice}) 초기화 실패.", traceback.format_exc()); sys.exit(1)
        elif backend_choice != "none" and not available_backends: self.ui.error(f"선택된 백엔드 '{backend_choice}' 모듈 없음."); sys.exit(1)
        else: self.ai = None; self.ui.console.print("[yellow]경고: AI 백엔드 'none', AI 기능 비활성.[/]")

    def _ensure_prereqs(self,cmd:str,needs_git:bool,needs_ai:bool):
        if needs_git and not self.git: self.ui.error(f"'{cmd}' 명령은 Git 저장소 내 실행 필요."); sys.exit(1)
        if needs_ai and not self.ai: self.ui.error(f"'{cmd}' 명령은 AI 백엔드 설정 필요. (--backend 옵션)"); sys.exit(1)

    def save(self):
        self._ensure_prereqs("save", True, True)
        try:
            default_task = self.git.get_current_branch() or (self.git.repo.head.commit.summary if self.git and self.git.repo else "작업 요약")
            task = self.ui.task_name(default=default_task)
            last_commit = self.git.get_last_state_commit()
            default_ctx = self.git.get_commit_messages_since(last_commit.hexsha if last_commit else None)
            context = self.ui.multiline("작업 내용 요약 (AI 요약 결과 붙여넣기 권장)", default=default_ctx)
            if not context.strip(): self.ui.error("요약 내용 없음, 저장 취소."); return
            self.state_dir.mkdir(exist_ok=True); self.art_dir.mkdir(exist_ok=True)
            artifacts = [f.name for f in self.art_dir.iterdir() if f.is_file()] if self.art_dir.exists() else []
            if self.art_dir.exists(): self.ui.console.print(f"[dim]현재 아티팩트 ({self.art_dir.relative_to(self.root_dir)}): {', '.join(artifacts) or '없음'}[/]")
            else: self.ui.console.print(f"[dim]아티팩트 폴더({self.art_dir.relative_to(self.root_dir)}) 없음.[/]")
            self.ui.console.print("\n[bold yellow]AI가 인수인계 문서 생성 중...[/]")
            md_summary = self.ai.make_summary(task, context, artifacts)
            self.ui.panel(md_summary, "AI 생성 요약본 (검증 전)")
            self.ui.console.print("[bold yellow]생성된 요약본 AI 검증 중...[/]")
            is_ok, val_msg = self.ai.verify_summary(md_summary)
            if not is_ok: raise RuntimeError(f"AI 생성 문서 검증 실패:\n{val_msg}")
            self.ui.notify("AI 검증 통과!", style="green")
            state_files, snap_dir = Serializer.save_state(md_summary, task, self.state_dir, self.art_dir, self.root_dir)
            commit_hash = self.git.save(state_files, task, snap_dir)
            self.ui.notify(f"상태 저장 완료! (Commit: {commit_hash})", style="bold green")
            html_f = next((f for f in state_files if f.name.endswith(".html")), None)
            if html_f and html_f.exists(): self.ui.console.print(f"[dim]HTML 프리뷰: {html_f.relative_to(self.root_dir)}[/]")
        except Exception as e: self.ui.error(f"Save 중 오류: {str(e)}", traceback.format_exc())

    def load(self, latest: bool = False):
        self._ensure_prereqs("load", True, True)
        try:
            states = self.git.list_states(self.state_dir)
            if not states: self.ui.error("저장된 상태 없음."); return
            sel_hash = states[-1]["hash"] if latest else self.ui.pick_state(states)
            if not sel_hash: return
            if latest: print(f"[info]최신 상태 로드: {sel_hash} ({next((s['task'] for s in states if s['hash']==sel_hash),'')})[/]")
            self.ui.console.print(f"[bold yellow]{sel_hash} 커밋에서 상태 로드 중...[/]")
            md_content = self.git.load_state(sel_hash, self.state_dir)
            prompt_fmt = Serializer.to_prompt(md_content, sel_hash)
            self.ui.panel(prompt_fmt, f"로드된 상태 (Commit: {sel_hash})", border_style="cyan")
            self.ui.console.print("[bold yellow]AI가 로드된 상태 분석/요약...[/]")
            report = self.ai.load_report(md_content)
            self.ui.panel(report, "AI 이해도 보고서", border_style="magenta")
        except Exception as e: self.ui.error(f"Load 중 오류: {str(e)}", traceback.format_exc())

    def diff(self, target: str = "HEAD"):
        self._ensure_prereqs("diff", True, False)
        try:
             self.ui.console.print(f"[bold yellow]'{target}' 대비 변경 사항 확인 중...[/]")
             diff_out = self.git.get_diff(target, color=True)
             self.ui.diff_panel(diff_out, target)
        except Exception as e: self.ui.error(f"Diff 중 오류: {str(e)}", traceback.format_exc())

    def verify_checksums(self, commit_hash: str):
        self._ensure_prereqs("verify", True, False)
        self.ui.console.print(f"[dim]커밋 {commit_hash}의 저장된 아티팩트 체크섬 표시. (실제 파일 비교는 미구현)[/]")
        try:
            commit = self.git.repo.commit(self.git.repo.git.rev_parse(commit_hash))
            meta_blob = None
            commit_state_dir_path_str = (self.state_dir.relative_to(self.root_dir)).as_posix()
            for item_path_str in [p.path for p in commit.tree.traverse()]:
                if item_path_str.startswith(commit_state_dir_path_str) and item_path_str.endswith(".meta.json"):
                    meta_blob = commit.tree[item_path_str]; break
            if not meta_blob: self.ui.error(f"커밋 {commit_hash} 메타파일 없음 ({commit_state_dir_path_str})."); return
            metadata = json.loads(meta_blob.data_stream.read().decode('utf-8'))
            checksums = metadata.get("artifact_checksums", {})
            if checksums: self.ui.panel(json.dumps(checksums,indent=2,ensure_ascii=False), f"저장된 체크섬 (Commit: {commit_hash})", border_style="magenta")
            else: print(f"[dim]커밋 {commit_hash}에 저장된 체크섬 정보 없음.[/]")
        except GitCommandError: self.ui.error(f"커밋 해시 '{commit_hash}' 없음.")
        except Exception as e: self.ui.error(f"체크섬 정보 로드 중 오류 ({commit_hash})", traceback.format_exc())

def main_cli_entry_point():
    global INITIAL_ROOT_PATH, STATE_DIR_MODULE_LEVEL, ART_DIR_MODULE_LEVEL # Use module-level for Serializer

    cli_root_path = pathlib.Path('.').resolve()
    is_git_repo_at_cli_root = False
    try:
        git_repo_root_obj = Repo('.', search_parent_directories=True)
        cli_root_path = pathlib.Path(git_repo_root_obj.working_tree_dir)
        is_git_repo_at_cli_root = True
    except InvalidGitRepositoryError: pass

    STATE_DIR_MODULE_LEVEL = cli_root_path / "ai_states" # These are used by Handover instance via self.state_dir
    ART_DIR_MODULE_LEVEL = cli_root_path / "artifacts"   # These are used by Handover instance via self.art_dir

    STATE_DIR_MODULE_LEVEL.mkdir(exist_ok=True) # Ensure they exist for direct use or by Handover
    ART_DIR_MODULE_LEVEL.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="AI 기반 프로젝트 인수인계 상태 관리 도구 (v1.1.5)", formatter_class=argparse.RawTextHelpFormatter)
    backend_choices = list(available_backends.keys()) if available_backends else []
    default_be = "none"
    if "ollama" in backend_choices: default_be = "ollama"
    elif backend_choices: default_be = backend_choices[0]
    parser.add_argument("--backend", default=os.getenv("AI_BACKEND", default_be), choices=backend_choices + ["none"], help=f"AI 백엔드 (기본값: 환경변수 AI_BACKEND 또는 {default_be}). 사용가능: {', '.join(backend_choices) or '없음'}. 'none'으로 AI 비활성화.")
    subparsers = parser.add_subparsers(dest="command", help="실행할 작업", required=True)
    cmd_configs = [("save", "현재 작업 상태를 요약/저장", True, True), ("load", "과거 저장된 상태 불러오기", True, True), ("diff", "현재 변경 사항 미리보기 (Git)", True, False), ("verify", "저장된 상태 아티팩트 체크섬 표시", True, False)]
    for name, help_txt, git_req, ai_req in cmd_configs:
        p = subparsers.add_parser(name, help=f"{help_txt}{' (Git 필요)' if git_req else ''}{' (AI 필요)' if ai_req else ''}")
        if name == "load": p.add_argument("-l", "--latest", action="store_true", help="가장 최근 상태 자동 로드")
        if name == "diff": p.add_argument("target", nargs="?", default="HEAD", help="비교 대상 Commit/Branch (기본값: HEAD)")
        if name == "verify": p.add_argument("commit", help="체크섬 정보를 확인할 상태 커밋 해시")
    args = parser.parse_args()

    chosen_cmd_config = next(c for c in cmd_configs if c[0] == args.command)
    if chosen_cmd_config[2] and not is_git_repo_at_cli_root: UI.error(f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재: {cli_root_path})"); sys.exit(1)
    if chosen_cmd_config[3] and (not available_backends or args.backend == "none"):
        msg = "사용 가능한 AI 백엔드가 없습니다. 'backends' 폴더 확인." if not available_backends else f"AI 백엔드가 '{args.backend}'(으)로 설정됨."
        UI.error(f"'{args.command}' 명령 실행 불가: {msg} AI 기능이 필요합니다."); sys.exit(1)

    print(f"[bold underline]Handover 스크립트 v1.1.5[/]")
    if is_git_repo_at_cli_root: print(f"[dim]프로젝트 루트: {cli_root_path}[/]")
    else: print(f"[dim]현재 작업 폴더: {cli_root_path} (Git 저장소 아님)[/]")

    try:
        handler = Handover(backend_choice=args.backend, current_root=cli_root_path)
        if args.command == "save": handler.save()
        elif args.command == "load": handler.load(latest=args.latest)
        elif args.command == "diff": handler.diff(target=args.target)
        elif args.command == "verify": handler.verify_checksums(commit_hash=args.commit)
    except Exception as e_handler: UI.error("핸들러 실행 중 예기치 않은 오류", traceback.format_exc()); sys.exit(1)

if __name__ == "__main__":
    if sys.version_info < (3, 8): print("[bold red]오류: Python 3.8 이상 필요.[/]"); sys.exit(1)
    main_cli_entry_point()