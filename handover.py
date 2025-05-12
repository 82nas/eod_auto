#!/usr/bin/env python3
# handover.py – 인수인계 v1.1.6 (AI 검증 디버깅 추가)

from __future__ import annotations
import os
import sys
import datetime
import json
import textwrap
import pathlib
import shutil
import traceback
import argparse
import hashlib
import importlib
import importlib.util  # 명시적 임포트 추가
from typing import List, Dict, Tuple, Optional, Type, Any
from dotenv import load_dotenv

# --- 의존성 로드 ---
try:
    # 필요한 클래스/예외 명시적으로 추가
    from git import (
        Repo,
        GitCommandError,
        InvalidGitRepositoryError,
        NoSuchPathError,
        Blob,
        Commit,
    )
    from rich import print, box
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.console import Console
    from rich.syntax import Syntax
    import markdown2
except ImportError as e:
    print(f"[bold red]오류: 필요한 라이브러리가 설치되지 않았습니다.[/]\n{e}")
    print(
        "팁: [yellow]pip install gitpython requests rich python-dotenv markdown2[/] 명령을 실행하세요."
    )
    sys.exit(1)

load_dotenv()

# backends 폴더는 handover.py 파일과 같은 디렉토리에 있다고 가정
BACKEND_DIR = pathlib.Path(__file__).resolve().parent / "backends"
COMMIT_TAG = "state("

# --- AI 백엔드 로딩 ---
AIBaseBackend = None  # 기본값 설정
try:
    base_spec = importlib.util.spec_from_file_location(
        "backends.base", BACKEND_DIR / "base.py"
    )
    if base_spec is None or base_spec.loader is None:
        raise ImportError(
            f"backends.base 모듈 스펙을 찾을 수 없습니다. 경로: {BACKEND_DIR / 'base.py'}"
        )
    backends_base_module = importlib.util.module_from_spec(base_spec)
    sys.modules["backends.base"] = backends_base_module  # sys.modules에 명시적으로 추가
    base_spec.loader.exec_module(backends_base_module)
    AIBaseBackend = backends_base_module.AIBaseBackend
except ImportError as e:
    print(f"[bold red]오류: backends.base 모듈 임포트 실패: {e}[/]")
except AttributeError:
    print("[bold red]오류: backends.base 모듈에서 AIBaseBackend 클래스 찾기 실패.[/]")
except FileNotFoundError:
    print(
        f"[bold red]오류: backends/base.py 파일을 찾을 수 없습니다. 경로: {BACKEND_DIR / 'base.py'}[/]"
    )

available_backends: Dict[str, Type[AIBaseBackend]] = {}
if AIBaseBackend and BACKEND_DIR.exists() and BACKEND_DIR.is_dir():
    for f_py in BACKEND_DIR.glob("*.py"):
        module_name_stem = f_py.stem
        if module_name_stem == "__init__" or module_name_stem == "base":
            continue
        try:
            full_module_name = f"backends.{module_name_stem}"
            spec = importlib.util.spec_from_file_location(full_module_name, f_py)
            if spec is None or spec.loader is None:
                print(f"[yellow]경고: 백엔드 모듈 스펙 로딩 실패 {f_py.name}[/]")
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_module_name] = module
            spec.loader.exec_module(module)

            for name, obj in module.__dict__.items():
                if (
                    isinstance(obj, type)
                    and AIBaseBackend
                    and issubclass(obj, AIBaseBackend)  # AIBaseBackend None 체크 추가
                    and obj is not AIBaseBackend
                ):
                    backend_name_from_class = obj.get_name()
                    if backend_name_from_class != "base":
                        available_backends[backend_name_from_class] = obj
        except ImportError as e:
            print(
                f"[yellow]경고: 백엔드 모듈 '{module_name_stem}' 로딩 중 임포트 오류: {e}[/]"
            )
        except AttributeError as e:
            print(
                f"[yellow]경고: 백엔드 클래스 '{name if 'name' in locals() else module_name_stem}' 속성 오류: {e}[/]"
            )
        except Exception as e:
            print(f"[yellow]경고: 백엔드 파일 '{f_py.name}' 처리 중 예외 발생: {e}[/]")
elif not AIBaseBackend:
    print("[yellow]경고: AIBaseBackend 로딩 실패로 백엔드 목록을 만들 수 없습니다.[/]")
else:
    print(
        f"[yellow]경고: 백엔드 디렉토리 '{BACKEND_DIR}'를 찾을 수 없거나 디렉토리가 아닙니다.[/]"
    )


# --- AIProvider 클래스 ---
class AIProvider:
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        if not AIBaseBackend:
            raise RuntimeError("AIProvider 초기화 실패: AIBaseBackend 로딩 실패.")
        if not available_backends and backend_name != "none":
            raise RuntimeError(
                "AIProvider 초기화 실패: 사용 가능한 AI 백엔드가 없습니다."
            )
        if backend_name == "none":
            self.backend = None
            return
        if backend_name not in available_backends:
            raise ValueError(
                f"알 수 없는 백엔드: '{backend_name}'. 사용 가능: {list(available_backends.keys()) + ['none']}"
            )

        BackendClass = available_backends[backend_name]
        try:
            self.backend: Optional[AIBaseBackend] = BackendClass(config)
        except Exception as e:
            print(f"[bold red]오류: 백엔드 '{backend_name}' 초기화 실패: {e}[/]")
            if hasattr(BackendClass, "get_config_description"):
                print(f"[yellow]필요 설정:\n{BackendClass.get_config_description()}[/]")
            raise e

    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        if not self.backend:
            raise RuntimeError(
                "AI 백엔드가 'none'으로 설정되어 요약을 생성할 수 없습니다."
            )
        return self.backend.make_summary(task, ctx, arts)

    # --- verify_summary 메서드 (디버깅 추가) ---
    def verify_summary(self, md: str) -> Tuple[bool, str]:
        """AI 생성 마크다운 검증 (백엔드 + 내부 구조 검사)"""
        if not self.backend:
            raise RuntimeError(
                "AI 백엔드가 'none'으로 설정되어 요약을 검증할 수 없습니다."
            )

        # --- 백엔드 호출 및 원시 결과 확인 ---
        try:
            # 실제 백엔드의 verify_summary 호출
            backend_is_ok, backend_msg = self.backend.verify_summary(md)
            # 디버깅 출력: 백엔드가 반환한 값 그대로 출력
            print("-" * 20 + " Backend Raw Response " + "-" * 20)  # 구분선 추가
            print(
                f"[DEBUG] Raw backend_is_ok: {backend_is_ok} (Type: {type(backend_is_ok)})"
            )
            print(f"[DEBUG] Raw backend_msg:\n{backend_msg}")
            print("-" * 50)
        except Exception as e_backend:
            # 백엔드 호출 자체에서 예외 발생 시
            print(f"[DEBUG] Backend verify_summary call failed: {e_backend}")
            # 즉시 실패로 처리하고 오류 메시지 반환
            return False, f"백엔드 검증 호출 실패: {e_backend}"
        # --- 원시 결과 확인 끝 ---

        # 최종 is_ok, msg 변수 초기화 (백엔드 결과 우선 사용)
        is_ok = backend_is_ok
        msg = backend_msg

        # --- 내부 구조 검증 (백엔드가 OK라고 했을 때만 진행) ---
        if is_ok:
            print(
                "[DEBUG] Backend reported OK. Starting internal structure checks..."
            )  # 내부 검증 시작 알림
            lines = md.strip().split("\n")
            headers = [line.strip() for line in lines if line.startswith("#")]
            required_headers_structure = [
                "#",
                "## 목표",
                "## 진행",
                "## 결정",
                "## 결과",
                "## 다음할일",
                "## 산출물",
            ]
            print(
                f"[DEBUG] Internal Check - Found Headers: {headers}"
            )  # 찾아낸 헤더 목록 출력

            # 1. 헤더 개수 검증
            if len(headers) != len(required_headers_structure):
                failure_reason = f"헤더 개수 불일치 (필수 {len(required_headers_structure)}개, 현재 {len(headers)}개)"
                print(
                    f"[DEBUG] Internal Check FAILED: {failure_reason}"
                )  # 실패 사유 출력
                is_ok = False  # 최종 상태 False로 변경
                msg = failure_reason  # 최종 메시지도 실패 사유로 변경
            else:
                # 2. 헤더 형식 및 순서 검증
                for i, expected_start_format in enumerate(required_headers_structure):
                    header_correct = False
                    current_header = headers[i]
                    # 첫 번째 헤더("# ") 형식 검사
                    if i == 0 and current_header.startswith("# "):
                        header_correct = True
                    # 나머지 헤더("## 이름 ") 형식 검사
                    elif i > 0 and (
                        current_header == expected_start_format
                        or current_header.startswith(expected_start_format + " ")
                    ):
                        header_correct = True

                    # 형식 검증 실패 시
                    if not header_correct:
                        if i == 0:
                            failure_reason = f"첫 번째 헤더 형식 오류: '{current_header}' (예상: '# 작업이름')"
                        else:
                            failure_reason = f"헤더 #{i+1} 형식 또는 순서 오류: '{current_header}' (예상: '{expected_start_format} 이름')"
                        print(
                            f"[DEBUG] Internal Check FAILED: {failure_reason}"
                        )  # 실패 사유 출력
                        is_ok = False  # 최종 상태 False로 변경
                        msg = failure_reason  # 최종 메시지도 실패 사유로 변경
                        break  # 첫 실패 지점에서 검증 중단
            if is_ok:  # 내부 검증 모두 통과 시
                print("[DEBUG] Internal structure checks PASSED.")
        # --- 내부 구조 검증 끝 ---
        else:
            # 백엔드가 False를 반환했으면 내부 검증은 건너뜀
            print("[DEBUG] Backend reported NOT OK. Skipping internal checks.")

        # 최종적으로 결정된 is_ok와 msg 값을 출력하고 반환
        print(
            f"[DEBUG] AIProvider.verify_summary FINAL return: is_ok={is_ok}, msg='{msg}'"
        )
        return is_ok, msg

    def load_report(self, md: str) -> str:
        if not self.backend:
            raise RuntimeError(
                "AI 백엔드가 'none'으로 설정되어 보고서를 로드할 수 없습니다."
            )
        return self.backend.load_report(md)


# --- GitRepo 클래스 ---
class GitRepo:
    def __init__(self, repo_path: pathlib.Path):
        self.repo: Optional[Repo] = None
        try:
            self.repo = Repo(
                repo_path, search_parent_directories=True
            )  # 상위 디렉토리 검색 추가
            if self.repo.working_dir != str(
                repo_path.resolve()
            ):  # 실제 찾은 경로가 입력 경로와 다르면 알려줌
                print(f"[dim]Git 저장소 루트 발견: {self.repo.working_dir}[/dim]")
            if self.repo.bare:
                print(
                    f"[yellow]경고: '{self.repo.working_dir}'는 bare 저장소입니다. 작업 디렉토리 분석이 제한될 수 있습니다.[/]"
                )
        except InvalidGitRepositoryError:
            pass  # Handover 생성자에서 None 처리
        except NoSuchPathError:
            print(f"[red]오류: Git 저장소 경로를 찾을 수 없습니다: {repo_path}[/]")
            pass  # Handover 생성자에서 None 처리
        except GitCommandError as e:
            print(
                f"[red]오류: Git 명령어 실행 실패 ({e.command}). Git 설치 및 PATH 확인 필요.[/]"
            )
            print(f"[dim]상세: {e.stderr}[/dim]")
            pass  # Handover 생성자에서 None 처리
        except Exception as e:
            print(f"[red]오류: Git 저장소 초기화 중 예외 발생: {e}[/]")
            pass  # Handover 생성자에서 None 처리

    def _safe(self, git_func, *args, **kwargs):
        if not self.repo:
            raise RuntimeError("GitRepo 객체가 초기화되지 않았습니다.")
        try:
            return git_func(*args, **kwargs)
        except GitCommandError as e:
            stderr = e.stderr.strip()
            raise RuntimeError(f"Git 명령어 실패: {e.command}\n오류: {stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Git 작업 중 예외 발생: {e}") from e

    def get_last_state_commit(self) -> Optional[Commit]:
        if not self.repo:
            return None
        try:
            for c in self.repo.iter_commits(max_count=200, first_parent=True):
                if c.message.startswith(COMMIT_TAG):
                    return c
        except Exception as e:
            print(f"[yellow]경고: 마지막 상태 커밋 검색 중 오류: {e}[/yellow]")
        return None

    def get_commit_messages_since(self, commit_hash: Optional[str]) -> str:
        if not self.repo:
            return "Git 저장소 없음."
        if not commit_hash:
            try:
                commits = list(self.repo.iter_commits(max_count=10, no_merges=True))
                log = "\n".join(
                    f"- {c.hexsha[:7]}: {c.summary}" for c in reversed(commits)
                )
                return (
                    f"최근 커밋 {len(commits)}개:\n{log}" if log else "최근 커밋 없음"
                )
            except Exception as e:
                return f"최근 커밋 로그 조회 실패: {e}"
        try:
            self.repo.commit(commit_hash)
            log_cmd = f"{commit_hash}..HEAD"
            commit_log = self.repo.git.log(
                log_cmd, "--pretty=format:- %h: %s", "--abbrev-commit", "--no-merges"
            )
            return (
                f"'{commit_hash[:8]}' 이후 커밋:\n{commit_log}"
                if commit_log
                else f"'{commit_hash[:8]}' 이후 커밋 없음"
            )
        except GitCommandError as e:
            return f"커밋 로그 조회 실패 ({commit_hash}): {e.stderr}"
        except Exception as e:
            return f"커밋 로그 조회 중 오류 ({commit_hash}): {e}"

    def get_current_branch(self) -> Optional[str]:
        if not self.repo:
            return "Git 저장소 없음"
        try:
            return self.repo.active_branch.name
        except TypeError:
            try:
                return f"DETACHED_HEAD@{self.repo.head.commit.hexsha[:7]}"
            except Exception:
                return "DETACHED_HEAD"
        except Exception as e:
            print(f"[yellow]경고: 현재 브랜치 확인 중 오류: {e}[/yellow]")
            return None

    def get_diff(self, target: str = "HEAD", color: bool = True) -> str:
        if not self.repo:
            return "Git 저장소 없음."
        try:
            self.repo.commit(target)
            color_opt = "--color=always" if color else "--color=never"
            staged_diff = self.repo.git.diff("--staged", target, color_opt)
            working_tree_vs_target_diff = self.repo.git.diff(target, color_opt)
            diff_output = ""
            has_staged = bool(staged_diff.strip())
            has_wt_vs_target = bool(working_tree_vs_target_diff.strip())
            if has_staged:
                diff_output += (
                    f"--- Staged Changes (vs {target}) ---\n{staged_diff}\n\n"
                )
            if has_wt_vs_target and working_tree_vs_target_diff != staged_diff:
                diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"
            elif not has_staged and has_wt_vs_target:
                diff_output += f"--- Changes in Working Directory (vs {target}) ---\n{working_tree_vs_target_diff}\n"
            return (
                diff_output.strip()
                if diff_output.strip()
                else f"'{target}'과(와) 변경 사항 없음 (Git 추적 파일 기준)"
            )
        except GitCommandError as e:
            return f"Diff 생성 Git 오류: {e.stderr}"
        except Exception as e:
            return f"Diff 생성 오류: {e}"

    def save(
        self,
        state_paths: List[pathlib.Path],
        task: str,
        snapshot_dir: Optional[pathlib.Path],
    ) -> str:
        if not self.repo:
            raise RuntimeError("Git 저장소가 없어 저장할 수 없습니다.")
        paths_to_add_rel = []
        repo_root = pathlib.Path(self.repo.working_dir)
        for p in state_paths:
            try:
                paths_to_add_rel.append(
                    p.resolve().relative_to(repo_root).as_posix()
                )  # Posix 경로로 변환
            except ValueError:
                print(f"[yellow]경고: Git 저장소 외부 경로 추가 시도 무시됨: {p}[/]")
        if snapshot_dir and snapshot_dir.exists() and any(snapshot_dir.iterdir()):
            try:
                paths_to_add_rel.append(
                    snapshot_dir.resolve().relative_to(repo_root).as_posix()
                )
            except ValueError:
                print(
                    f"[yellow]경고: Git 저장소 외부 스냅샷 경로 추가 시도 무시됨: {snapshot_dir}[/]"
                )

        if not paths_to_add_rel:
            raise ValueError(
                "저장할 상태 파일 또는 스냅샷 파일이 없습니다 (Git 저장소 내)."
            )

        # git add 실행
        self._safe(self.repo.git.add, *paths_to_add_rel)

        commit_msg = f"{COMMIT_TAG}{task})"
        commit_hash = ""
        try:
            # git commit 실행
            self._safe(self.repo.index.commit, commit_msg)
            commit_hash = self.repo.head.commit.hexsha[:8]  # 성공 시 커밋 해시
        except RuntimeError as e:
            err_str = str(e).lower()
            if (
                "nothing to commit" in err_str
                or "no changes added to commit" in err_str
                or "changes not staged for commit" in err_str
            ):
                print(
                    "[yellow]경고: 커밋할 변경 사항이 없습니다. 이전 상태와 동일할 수 있습니다.[/]"
                )
                commit_hash = (
                    self.repo.head.commit.hexsha[:8] + " (변경 없음)"
                )  # 현재 HEAD 커밋 해시 사용
            else:
                raise e  # 다른 RuntimeError는 다시 발생시킴

        # Push 로직
        if self.repo.remotes:
            try:
                current_branch_name = self.get_current_branch()
                if current_branch_name and not current_branch_name.startswith(
                    "DETACHED_HEAD"
                ):
                    print(
                        f"[dim]'{current_branch_name}' 브랜치를 원격 저장소(origin)에 푸시 시도...[/]"
                    )
                    self._safe(self.repo.git.push, "origin", current_branch_name)
                    print("[green]원격 저장소에 푸시 완료.[/]")
                else:
                    print(
                        f"[yellow]경고: 현재 브랜치({current_branch_name})가 특정되지 않았거나 Detached HEAD 상태이므로 푸시를 건너뜁니다.[/]"
                    )
            except RuntimeError as e:
                print(
                    f"[yellow]경고: 원격 저장소 푸시 실패. 로컬에는 커밋되었습니다. ({e})[/]"
                )
            except Exception as e_general:
                print(
                    f"[yellow]경고: 원격 저장소 푸시 중 예기치 않은 오류: {e_general}[/]"
                )
        else:
            print("[yellow]경고: 설정된 원격 저장소가 없어 푸시를 건너뜁니다.[/]")

        return commit_hash  # 최종 커밋 해시 반환

    def _get_relative_path_str(self, target_path: pathlib.Path) -> Optional[str]:
        if not self.repo:
            return None
        try:
            return target_path.resolve().relative_to(self.repo.working_dir).as_posix()
        except ValueError:
            return None
        except Exception:
            return None

    def list_states(self, current_app_state_dir: pathlib.Path) -> List[Dict]:
        if not self.repo:
            return []
        items = []
        search_rel_path_str = self._get_relative_path_str(current_app_state_dir)
        if not search_rel_path_str:
            print(
                f"[yellow]경고: 상태 디렉토리({current_app_state_dir})가 Git 저장소 내에 없어 상태 목록을 검색할 수 없습니다.[/]"
            )
            return []
        try:
            commits = list(
                self.repo.iter_commits(
                    max_count=100, first_parent=True, paths=search_rel_path_str
                )
            )
        except Exception as e:
            print(
                f"[yellow]경고: 특정 경로({search_rel_path_str}) 커밋 검색 실패, 전체 커밋에서 검색 시도: {e}[/]"
            )
            # 전체 커밋 검색 시도는 너무 많은 결과를 반환할 수 있으므로 주의, 여기서는 일단 제외
            # commits = list(self.repo.iter_commits(max_count=100, first_parent=True))
            return []  # 특정 경로 검색 실패 시 빈 목록 반환

        for c in commits:
            if not c.message.startswith(COMMIT_TAG):
                continue
            headline = ""
            meta_blob = None
            try:
                for item in c.tree.traverse():
                    if (
                        isinstance(item, Blob)
                        and item.path.startswith(search_rel_path_str)
                        and item.path.endswith(".meta.json")
                    ):
                        meta_blob = item
                        break
                if meta_blob:
                    metadata = json.loads(meta_blob.data_stream.read().decode("utf-8"))
                    headline = metadata.get("headline", "")
            except Exception as e:
                headline = f"[메타데이터 오류: {e}]"
            items.append(
                {
                    "hash": c.hexsha[:8],
                    "task": c.message[len(COMMIT_TAG) : -1].strip(),
                    "time": datetime.datetime.fromtimestamp(c.committed_date).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "head": headline or "-",
                }
            )
        return list(reversed(items))

    def load_state(self, commit_hash: str, current_app_state_dir: pathlib.Path) -> str:
        if not self.repo:
            raise RuntimeError("Git 저장소가 없어 로드할 수 없습니다.")
        try:
            commit_obj = self.repo.commit(self.repo.git.rev_parse(commit_hash))
        except Exception as e:
            raise RuntimeError(f"커밋 '{commit_hash}' 접근 오류: {e}") from e

        search_rel_path_str = self._get_relative_path_str(current_app_state_dir)
        if not search_rel_path_str:
            raise RuntimeError(
                f"상태 디렉토리({current_app_state_dir})가 Git 저장소 내에 없습니다."
            )
        try:
            for item in commit_obj.tree.traverse():
                if (
                    isinstance(item, Blob)
                    and item.path.startswith(search_rel_path_str)
                    and item.path.endswith(".md")
                ):
                    return item.data_stream.read().decode("utf-8")
            raise RuntimeError(
                f"커밋 '{commit_hash}' 내 경로 '{search_rel_path_str}'에서 상태 파일(.md)을 찾을 수 없습니다."
            )
        except Exception as e:
            raise RuntimeError(
                f"커밋 '{commit_hash}' 상태 로드 중 예기치 않은 오류: {e}"
            )


# --- Serializer 클래스 ---
class Serializer:
    @staticmethod
    def _calculate_sha256(fp: pathlib.Path) -> Optional[str]:
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
            return None
        except Exception:
            return None

    @staticmethod
    def _generate_html(md: str, title: str) -> str:
        css = """<style>body{font-family:sans-serif;line-height:1.6;padding:20px;max-width:800px;margin:auto;color:#333}h1,h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.5em;margin-bottom:1em}h1{font-size:2em}h2{font-size:1.5em}ul,ol{padding-left:2em}li{margin-bottom:.5em}code{background-color:#f0f0f0;padding:.2em .4em;border-radius:3px;font-family:monospace;font-size:.9em}pre{background-color:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}pre code{background-color:transparent;padding:0;border-radius:0}blockquote{border-left:4px solid #ccc;padding-left:1em;color:#666;margin-left:0}table{border-collapse:collapse;width:100%;margin-bottom:1em}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}</style>"""
        try:
            body = markdown2.markdown(
                md,
                extras=[
                    "metadata",
                    "fenced-code-blocks",
                    "tables",
                    "strike",
                    "task_list",
                    "code-friendly",
                    "markdown-in-html",
                ],
            )
            title_meta = title
            if hasattr(body, "metadata") and body.metadata.get("title"):
                title_meta = body.metadata["title"]
            return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{textwrap.shorten(title_meta, width=50, placeholder="...")}</title>{css}</head><body>{body}</body></html>"""
        except Exception as e:
            print(f"[yellow]경고: Markdown -> HTML 변환 중 오류 발생: {e}[/]")
            escaped_md = "".join(
                (
                    c
                    if c.isalnum() or c in " .,;:!?/\\#$%&'()*+-=<>[]_{}|`~"
                    else f"&#{ord(c)};"
                )
                for c in md
            )
            return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>HTML 생성 오류</title></head><body><h1>HTML 생성 오류</h1><p>Markdown 내용을 표시하는 데 문제가 발생했습니다:</p><pre>{escaped_md}</pre></body></html>"""

    @staticmethod
    def save_state(
        md: str,
        task: str,
        current_app_state_dir: pathlib.Path,
        current_app_art_dir: pathlib.Path,
        current_app_root: pathlib.Path,
    ) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        safe_task = (
            "".join(c for c in task if c.isalnum() or c in (" ", "_", "-"))
            .strip()
            .replace(" ", "_")
        )
        if not safe_task:
            safe_task = "untitled_task"
        base_fn = f"{ts}_{safe_task}"
        try:  # 디렉토리 생성 시 권한 문제 등 예외 처리
            current_app_state_dir.mkdir(parents=True, exist_ok=True)
            current_app_art_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"상태/아티팩트 디렉토리 생성 실패: {e}") from e

        state_f = current_app_state_dir / f"{base_fn}.md"
        html_f = current_app_state_dir / f"{base_fn}.html"
        meta_f = current_app_state_dir / f"{base_fn}.meta.json"

        try:
            state_f.write_text(md, encoding="utf-8")
        except IOError as e:
            raise RuntimeError(f"MD 파일 저장 실패 ({state_f}): {e}") from e

        html_content = Serializer._generate_html(md, task)
        html_ok = False
        if html_content:
            try:
                html_f.write_text(html_content, encoding="utf-8")
                html_ok = True
            except IOError as e:
                print(f"[yellow]경고: HTML 파일 저장 실패 ({html_f.name}): {e}[/]")
            except Exception as e:
                print(f"[yellow]경고: HTML 파일 저장 중 예외 ({html_f.name}): {e}[/]")

        snap_dir = None
        checksums = {}
        if current_app_art_dir.exists() and current_app_art_dir.is_dir():
            arts = [f for f in current_app_art_dir.iterdir() if f.is_file()]
            if arts:
                snapshot_sub_dir_name = f"{base_fn}_artifacts"
                snap_dir = current_app_art_dir / snapshot_sub_dir_name
                try:
                    snap_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    print(
                        f"[yellow]경고: 스냅샷 디렉토리 생성 실패 ({snap_dir}): {e}[/]"
                    )
                    snap_dir = None  # 생성 실패 시 스냅샷 진행 불가

                if snap_dir:  # 스냅샷 폴더 생성 성공 시 진행
                    print(
                        f"[dim]아티팩트 스냅샷 ({len(arts)}개) -> '{snap_dir.relative_to(current_app_root)}'[/]"
                    )
                    for f_art in arts:
                        try:
                            target_path = snap_dir / f_art.name
                            shutil.copy2(f_art, target_path)
                            cs = Serializer._calculate_sha256(target_path)
                            if cs:
                                checksums[f_art.name] = cs
                        except Exception as copy_e:
                            print(
                                f"[yellow]경고: 아티팩트 파일 '{f_art.name}' 복사/해시 실패: {copy_e}[/]"
                            )
            # else: print("[dim]아티팩트 폴더에 파일이 없어 스냅샷을 건너<0xEB><0x9B><0x81>니다.[/]") # 아티팩트 없으면 조용히 넘어감
        # else: print(f"[dim]아티팩트 폴더({current_app_art_dir})가 없거나 디렉토리가 아닙니다.[/]") # 폴더 없어도 조용히

        headline = task
        for ln in md.splitlines():
            ln_s = ln.strip()
            if ln_s.startswith("#"):
                headline = ln_s.lstrip("# ").strip()
                break

        meta = {
            "task": task,
            "ts": ts,
            "headline": headline,
            "artifact_checksums": checksums,
        }
        try:
            meta_f.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except IOError as e:
            raise RuntimeError(f"메타데이터 파일 저장 실패 ({meta_f}): {e}") from e
        except Exception as e:
            raise RuntimeError(f"메타데이터 JSON 생성/저장 실패: {e}") from e

        paths_to_commit = [state_f, meta_f]
        if html_ok and html_f.exists():
            paths_to_commit.append(html_f)
        valid_snap_dir = (
            snap_dir
            if (snap_dir and snap_dir.exists() and any(snap_dir.iterdir()))
            else None
        )
        return paths_to_commit, valid_snap_dir

    @staticmethod
    def to_prompt(md: str, commit: str) -> str:
        return f"### 이전 상태 (Commit: {commit}) ###\n\n{md}\n\n### 상태 정보 끝 ###"


# --- UI 클래스 ---
class UI:
    console = Console()

    @staticmethod
    def task_name(default: str = "작업 요약") -> str:
        return Prompt.ask("[bold cyan]작업 이름[/]", default=default)

    @staticmethod
    def multiline(label: str, default: str = "") -> str:
        UI.console.print(f"\n[bold cyan]{label}[/]")
        UI.console.print("[dim](입력 완료: 빈 줄에서 Enter 두 번)[/]")
        lines = []
        if default:
            default_preview = (
                textwrap.shorten(default.splitlines()[0], width=100, placeholder="...")
                if default
                else ""
            )
            if len(default.splitlines()) > 1:
                default_preview += " (...)"
            print(
                Panel(
                    default_preview,
                    title="[dim]자동 제안 (편집 가능, 전체 내용은 아래 입력)[/]",
                    border_style="dim",
                    expand=False,
                )
            )
        blank_count = 0
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                blank_count += 1
            else:
                blank_count = 0
            if blank_count >= 2:
                break
            lines.append(line)
        final = "\n".join(lines).strip()
        if not final and default:
            print("[dim]입력 없음, 자동 제안 내용을 사용합니다.[/]")
            return default
        return final

    @staticmethod
    def notify(msg: str, style: str = "green"):
        UI.console.print(f"\n[bold {style}]✔ {msg}[/]")

    @staticmethod
    def error(msg: str, details: Optional[str] = None):
        UI.console.print(f"\n[bold red]❌ 오류: {msg}[/]")
        if details:
            details_lines = details.strip().splitlines()
            max_lines = 15
            display_details = "\n".join(details_lines[:max_lines])
            if len(details_lines) > max_lines:
                display_details += "\n[dim]... (이하 생략)[/dim]"
            UI.console.print(
                Panel(
                    display_details,
                    title="[dim]상세 정보 (Traceback)[/]",
                    border_style="dim red",
                    expand=False,
                )
            )

    @staticmethod
    def pick_state(states: List[Dict]) -> Optional[str]:
        if not states:
            print("[yellow]저장된 상태가 없습니다.[/]")
            return None
        tb = Table(
            title="[bold]저장된 인수인계 상태 목록[/]",
            box=box.ROUNDED,
            show_lines=True,
            expand=False,
        )
        tb.add_column("#", style="dim", justify="right", width=3)
        tb.add_column("커밋", style="cyan", no_wrap=True, width=10)
        tb.add_column("작업", style="magenta", min_width=20, overflow="fold")
        tb.add_column("시각", style="green", no_wrap=True, width=18)
        tb.add_column("헤드라인", style="yellow", overflow="fold", min_width=30)
        for i, s in enumerate(states):
            tb.add_row(str(i), s["hash"], s["task"], s["time"], s["head"])
        UI.console.print(tb)
        choices = [str(i) for i in range(len(states))]
        sel = Prompt.ask(
            "[bold cyan]로드할 상태 번호 (취소하려면 Enter)[/]",
            choices=choices + [""],
            default="",
            show_choices=False,
        )
        if sel.isdigit() and 0 <= int(sel) < len(states):
            selected_state = states[int(sel)]
            print(
                f"[info]선택된 커밋: {selected_state['hash']} (작업: {selected_state['task']})[/]"
            )
            return selected_state["hash"]
        print("[info]상태 로드를 취소했습니다.[/]")
        return None

    @staticmethod
    def panel(txt: str, title: str, border_style: str = "blue"):
        UI.console.print(
            Panel(
                txt,
                title=f"[bold]{title}[/]",
                border_style=border_style,
                expand=False,
                padding=(1, 2),
            )
        )

    @staticmethod
    def diff_panel(txt: str, target: str):
        if not txt or "변경 사항 없음" in txt:
            print(f"[dim]{txt}[/]")
            return
        if txt.startswith("Diff 생성 오류") or txt.startswith("Diff 생성 Git 오류"):
            print(f"[red]{txt}[/]")
            return
        try:
            syntax_obj = Syntax(
                txt, "diff", theme="default", line_numbers=False, word_wrap=False
            )
            UI.console.print(
                Panel(
                    syntax_obj,
                    title=f"[bold]Diff (vs {target})[/]",
                    border_style="yellow",
                    expand=True,
                )
            )
        except Exception as e:
            print(f"[red]Diff 출력 중 오류 발생: {e}[/]")
            print(txt)


# --- Handover 클래스 ---
class Handover:
    def __init__(self, backend_choice: str, current_app_root: pathlib.Path):
        self.ui = UI()
        self.app_root = current_app_root
        self.state_dir = self.app_root / "ai_states"
        self.art_dir = self.app_root / "artifacts"
        self.git: Optional[GitRepo] = None
        self.ai: Optional[AIProvider] = None

        # GitRepo 초기화 시도 (실패해도 self.git은 None 유지)
        try:
            git_repo_candidate = GitRepo(self.app_root)
            if git_repo_candidate.repo:  # 초기화 성공 시 할당
                self.git = git_repo_candidate
        except Exception as e:
            self.ui.error(f"GitRepo 객체 생성 실패: {e}", traceback.format_exc())

        # AIProvider 초기화 시도
        if backend_choice != "none":
            if available_backends:
                try:
                    self.ai = AIProvider(backend_name=backend_choice, config={})
                except Exception:
                    self.ui.error(
                        f"AIProvider ('{backend_choice}') 설정 실패. AI 기능 비활성화됨.",
                        traceback.format_exc(),
                    )
            else:
                self.ui.error(
                    f"AI 백엔드 '{backend_choice}' 로드 불가 (사용 가능한 백엔드 없음). AI 기능 비활성화됨."
                )
        # backend_choice == "none" 이면 self.ai는 None 유지

    def _ensure_prereqs(self, cmd: str, needs_git: bool, needs_ai: bool):
        if needs_git and not self.git:
            self.ui.error(f"'{cmd}' 명령은 Git 저장소 내에서 실행해야 합니다.")
            sys.exit(1)
        if needs_ai and not self.ai:
            self.ui.error(
                f"'{cmd}' 명령은 AI 백엔드가 설정되어야 합니다. ('--backend' 옵션 확인)"
            )
            sys.exit(1)

    def save(self):
        self._ensure_prereqs("save", True, True)  # AI와 Git 모두 필요
        try:
            default_task_name = "작업 요약"
            if self.git and self.git.repo:  # GitRepo 및 내부 repo 객체 유효성 확인
                try:
                    current_branch = self.git.get_current_branch()
                    if current_branch and not current_branch.startswith(
                        "DETACHED_HEAD"
                    ):
                        default_task_name = current_branch
                    elif self.git.repo.head.is_valid():
                        default_task_name = self.git.repo.head.commit.summary
                except Exception as e_git_info:
                    print(
                        f"[yellow]경고: 기본 작업 이름 가져오기 실패 ({e_git_info})[/]"
                    )
            task_name_input = self.ui.task_name(default=default_task_name)

            default_context_summary = "작업 내용을 입력하세요."
            if self.git:  # GitRepo 객체 유효성 확인
                last_saved_commit = self.git.get_last_state_commit()
                default_context_summary = self.git.get_commit_messages_since(
                    last_saved_commit.hexsha if last_saved_commit else None
                )
            context_summary_input = self.ui.multiline(
                "작업 내용 요약 (AI가 생성한 요약을 붙여넣거나 직접 작성)",
                default=default_context_summary,
            )
            if not context_summary_input.strip():
                self.ui.error("작업 내용 요약이 비어있어 저장을 취소합니다.")
                return

            current_artifacts = []
            if self.art_dir.exists() and self.art_dir.is_dir():
                current_artifacts = [
                    f.name for f in self.art_dir.iterdir() if f.is_file()
                ]
                self.ui.console.print(
                    f"[dim]현재 아티팩트 ({self.art_dir.relative_to(self.app_root)}): {', '.join(current_artifacts) or '없음'}[/]"
                )
            else:
                self.ui.console.print(
                    f"[yellow]경고: 아티팩트 폴더({self.art_dir.relative_to(self.app_root)})를 찾을 수 없습니다.[/]"
                )

            self.ui.console.print(
                "\n[bold yellow]AI가 인수인계 문서를 생성 중입니다...[/]"
            )
            # self.ai는 _ensure_prereqs에서 None 아님을 보장
            generated_markdown = self.ai.make_summary(
                task_name_input, context_summary_input, current_artifacts
            )
            self.ui.panel(generated_markdown, "AI 생성 요약본 (검증 전)")

            self.ui.console.print(
                "[bold yellow]생성된 요약본을 AI가 검증 중입니다...[/]"
            )
            # === AI 검증 호출 (디버깅 추가됨) ===
            is_valid_summary, validation_message = self.ai.verify_summary(
                generated_markdown
            )

            # === 검증 결과 처리 ===
            if not is_valid_summary:
                # 실패 시 RuntimeError 발생 (디버깅 통해 원인 파악 후 이 부분 조정 가능)
                raise RuntimeError(
                    f"AI가 생성한 인수인계 문서 검증 실패:\n{validation_message}"
                )
            else:
                # 성공 시 알림
                self.ui.notify("AI 검증 통과!", style="green")

            # Serializer.save_state 호출
            saved_state_files, artifact_snapshot_dir = Serializer.save_state(
                generated_markdown,
                task_name_input,
                self.state_dir,
                self.art_dir,
                self.app_root,
            )

            # Git 저장 (self.git은 _ensure_prereqs에서 None 아님을 보장)
            commit_short_hash = self.git.save(
                saved_state_files, task_name_input, artifact_snapshot_dir
            )
            self.ui.notify(
                f"인수인계 상태 저장 완료! (Commit: {commit_short_hash})",
                style="bold green",
            )

            generated_html_file = next(
                (f for f in saved_state_files if f.name.endswith(".html")), None
            )
            if generated_html_file and generated_html_file.exists():
                self.ui.console.print(
                    f"[dim]HTML 프리뷰 생성됨: {generated_html_file.relative_to(self.app_root)}[/]"
                )

        except Exception as e:
            self.ui.error(f"Save 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def load(self, latest: bool = False):
        self._ensure_prereqs("load", True, True)  # AI와 Git 모두 필요
        try:
            # self.git은 None 아님 보장
            saved_states_list = self.git.list_states(self.state_dir)
            if not saved_states_list:
                self.ui.error("저장된 인수인계 상태가 없습니다.")
                return

            selected_commit_hash: Optional[str]
            if latest:
                if not saved_states_list:
                    self.ui.error("저장된 상태가 없어 최근 상태를 로드할 수 없습니다.")
                    return
                selected_commit_hash = saved_states_list[-1]["hash"]
                print(
                    f"[info]가장 최근 상태 로드 중: {selected_commit_hash} (작업명: {saved_states_list[-1]['task']})[/]"
                )
            else:
                selected_commit_hash = self.ui.pick_state(saved_states_list)

            if not selected_commit_hash:
                return

            self.ui.console.print(
                f"[bold yellow]{selected_commit_hash} 커밋에서 상태 정보를 로드 중입니다...[/]"
            )
            markdown_content = self.git.load_state(selected_commit_hash, self.state_dir)
            prompt_formatted_content = Serializer.to_prompt(
                markdown_content, selected_commit_hash
            )
            self.ui.panel(
                prompt_formatted_content,
                f"로드된 상태 (Commit: {selected_commit_hash})",
                border_style="cyan",
            )

            self.ui.console.print(
                "[bold yellow]AI가 로드된 상태를 분석하고 이해도를 보고합니다...[/]"
            )
            # self.ai는 None 아님 보장
            ai_report = self.ai.load_report(markdown_content)
            self.ui.panel(ai_report, "AI 이해도 보고서", border_style="magenta")

        except Exception as e:
            self.ui.error(f"Load 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def diff(self, target: str = "HEAD"):
        self._ensure_prereqs("diff", True, False)  # Git 필요, AI 불필요
        try:
            self.ui.console.print(
                f"[bold yellow]'{target}' 대비 현재 변경 사항을 확인 중입니다... (Git 추적 파일 기준)[/]"
            )
            # self.git은 None 아님 보장
            diff_output_text = self.git.get_diff(target, color=True)
            self.ui.diff_panel(diff_output_text, target)
        except Exception as e:
            self.ui.error(f"Diff 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def verify_checksums(self, commit_hash: str):
        self._ensure_prereqs("verify", True, False)  # Git 필요, AI 불필요
        self.ui.console.print(
            f"[dim]커밋 {commit_hash}의 저장된 아티팩트 체크섬 정보를 표시합니다. (실제 파일 비교 검증은 아직 구현되지 않았습니다.)[/]"
        )
        # self.git은 None 아님 보장
        try:
            commit_obj = self.git.repo.commit(self.git.repo.git.rev_parse(commit_hash))
            meta_blob: Optional[Blob] = None
            state_dir_rel_path_str = self.git._get_relative_path_str(self.state_dir)
            if not state_dir_rel_path_str:
                self.ui.error(
                    f"상태 디렉토리({self.state_dir})가 Git 저장소 내에 없어 메타데이터를 찾을 수 없습니다."
                )
                return

            for item in commit_obj.tree.traverse():
                if (
                    isinstance(item, Blob)
                    and item.path.startswith(state_dir_rel_path_str)
                    and item.path.endswith(".meta.json")
                ):
                    meta_blob = item
                    break

            if not meta_blob:
                self.ui.error(
                    f"커밋 {commit_hash}에서 메타데이터 파일(.meta.json)을 찾을 수 없습니다. (탐색 경로: {state_dir_rel_path_str})"
                )
                return

            metadata_content = json.loads(meta_blob.data_stream.read().decode("utf-8"))
            artifact_checksums_data = metadata_content.get("artifact_checksums", {})
            if artifact_checksums_data:
                checksums_pretty_str = json.dumps(
                    artifact_checksums_data, indent=2, ensure_ascii=False
                )
                self.ui.panel(
                    checksums_pretty_str,
                    f"저장된 아티팩트 체크섬 (Commit: {commit_hash})",
                    border_style="magenta",
                )
            else:
                print(
                    f"[dim]커밋 {commit_hash}에 저장된 아티팩트 체크섬 정보가 없습니다.[/]"
                )
        except GitCommandError as e:
            self.ui.error(
                f"Git 오류: 유효한 커밋 해시가 아니거나 찾을 수 없습니다 ('{commit_hash}'). {e.stderr}"
            )
        except Exception as e:
            self.ui.error(
                f"체크섬 정보 로드/표시 중 오류 ({commit_hash}): {str(e)}",
                traceback.format_exc(),
            )


# --- 스크립트 진입점 ---
def main_cli_entry_point():
    cli_root_path = pathlib.Path(".").resolve()
    git_repo_root_obj = None
    is_git_repo_at_cli_root = False
    try:
        git_repo_root_obj = Repo(cli_root_path, search_parent_directories=True)
        found_root_path = pathlib.Path(git_repo_root_obj.working_tree_dir)
        if (
            cli_root_path.resolve() == found_root_path.resolve()
            or cli_root_path in found_root_path.parents
        ):
            cli_root_path = found_root_path  # Git 루트를 기준으로 경로 설정
            is_git_repo_at_cli_root = True
    except InvalidGitRepositoryError:
        pass
    except Exception as e:
        print(
            f"[yellow]경고: Git 저장소 확인 중 오류 발생 (Git 기능 사용 불가): {e}[/]"
        )

    app_state_dir = cli_root_path / "ai_states"
    app_art_dir = cli_root_path / "artifacts"
    try:
        app_state_dir.mkdir(parents=True, exist_ok=True)
        app_art_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"[red]오류: 필수 디렉토리 생성 실패 ({app_state_dir} 또는 {app_art_dir}): {e}[/]"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="AI 기반 프로젝트 인수인계 상태 관리 도구 (v1.1.6 - Debug)",
        formatter_class=argparse.RawTextHelpFormatter,
    )  # 버전명에 Debug 추가
    backend_choices_list = list(available_backends.keys()) if available_backends else []
    default_be = "none"
    if "ollama" in backend_choices_list:
        default_be = "ollama"
    elif backend_choices_list:
        default_be = backend_choices_list[0]

    parser.add_argument(
        "--backend",
        default=os.getenv("AI_BACKEND", default_be),
        choices=backend_choices_list + ["none"],
        help=f"AI 백엔드 (기본값: 환경변수 AI_BACKEND 또는 '{default_be}'). 사용가능: {', '.join(backend_choices_list) or '없음'}. 'none'으로 AI 비활성화.",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="실행할 작업", required=True
    )
    # 도움말에서 중복 제거
    cmd_configs = [
        ("save", "현재 작업 상태를 요약/저장", True, True),
        ("load", "과거 저장된 상태 불러오기", True, True),
        ("diff", "현재 변경 사항 미리보기", True, False),  # 중복 "(Git)" 제거
        ("verify", "저장된 상태 아티팩트 체크섬 표시", True, False),
    ]
    for name, help_txt, git_req, ai_req in cmd_configs:
        p = subparsers.add_parser(
            name,
            help=f"{help_txt}{' (Git 필요)' if git_req else ''}{' (AI 필요)' if ai_req else ''}",
        )
        if name == "load":
            p.add_argument(
                "-l", "--latest", action="store_true", help="가장 최근 상태 자동 로드"
            )
        if name == "diff":
            p.add_argument(
                "target",
                nargs="?",
                default="HEAD",
                help="비교 대상 Commit/Branch (기본값: HEAD)",
            )
        if name == "verify":
            p.add_argument("commit", help="체크섬 정보를 확인할 상태 커밋 해시")

    args = parser.parse_args()

    chosen_cmd_config = next((c for c in cmd_configs if c[0] == args.command), None)
    if not chosen_cmd_config:
        UI.error(f"알 수 없는 명령어: {args.command}")
        sys.exit(1)
    _, _, git_needed, ai_needed = chosen_cmd_config

    if git_needed and not is_git_repo_at_cli_root:
        UI.error(
            f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 위치는 Git 저장소가 아님: {cli_root_path})"
        )
        sys.exit(1)
    if ai_needed:
        if args.backend == "none":
            UI.error(
                f"'{args.command}' 명령 실행 불가: AI 기능이 필요하지만 '--backend' 옵션이 'none'으로 설정되었습니다."
            )
            sys.exit(1)
        elif args.backend not in available_backends:
            UI.error(
                f"'{args.command}' 명령 실행 불가: 선택된 AI 백엔드 '{args.backend}'를 로드할 수 없습니다. 'backends' 폴더 또는 백엔드 설정을 확인하세요."
            )
            sys.exit(1)

    print("[bold underline]Handover 스크립트 v1.1.6 (Debug Mode)[/]")  # 버전명 수정
    if is_git_repo_at_cli_root:
        print(f"[dim]프로젝트 루트 (Git): {cli_root_path}[/]")
    else:
        print(f"[dim]현재 작업 폴더 (Git 저장소 아님): {cli_root_path}[/]")

    if args.backend != "none" and args.backend in available_backends:
        print(f"[dim]AI 백엔드 사용: [bold cyan]{args.backend}[/][/dim]")
    elif args.backend == "none" and not ai_needed:
        print("[dim]AI 백엔드: [bold yellow]none (비활성화됨)[/][/dim]")

    try:
        handler = Handover(backend_choice=args.backend, current_app_root=cli_root_path)
        if args.command == "save":
            handler.save()
        elif args.command == "load":
            handler.load(latest=args.latest)
        elif args.command == "diff":
            handler.diff(target=args.target)
        elif args.command == "verify":
            handler.verify_checksums(commit_hash=args.commit)
    except Exception:
        UI.error("핸들러 실행 중 예기치 않은 오류", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("[bold red]오류: Python 3.8 이상 필요.[/]")
        sys.exit(1)
    main_cli_entry_point()
