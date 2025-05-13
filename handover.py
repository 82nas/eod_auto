#!/usr/bin/env python3
# handover.py – 인수인계 v1.1.12 (System Instruction 내장 및 AIProvider.make_summary 개선)

from __future__ import annotations
import os
import sys
import datetime
import json
# import textwrap # dedent를 직접 임포트하므로 전체 textwrap 임포트는 불필요할 수 있음
from textwrap import dedent # dedent 직접 임포트
import pathlib
import shutil
import argparse
import hashlib
import importlib
import importlib.util
import subprocess
import re
import traceback
from typing import List, Dict, Tuple, Optional, Type, Any
from dotenv import load_dotenv

# --- 의존성 로드 ---
try:
    from git import Repo, GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Blob, Commit, HookExecutionError
    # import requests # 현재 코드에서 직접 사용되지 않는 것으로 보임 (AI 백엔드에서 사용)
    from rich import print, box
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.console import Console
    from rich.syntax import Syntax
    import markdown2 # type: ignore
except ImportError as e:
    print(f"[bold red]오류: 필요한 라이브러리가 설치되지 않았습니다.[/]\n{e}")
    print("팁: [yellow]pip install gitpython rich python-dotenv markdown2[/] 명령을 실행하세요. (requests는 필요시 AI 백엔드에 설치)")
    sys.exit(1)

load_dotenv()

BACKEND_DIR = pathlib.Path(__file__).resolve().parent / "backends"
COMMIT_TAG = "state("

# ----------------------------------------------------------------------
#  System Instruction (Markdown) – 사용자 제공 Self-Filling Prompt
# ----------------------------------------------------------------------
SYSTEM_INSTRUCTION_MD = dedent(
    """
# ✨ AI Handover-Prompt (Self-Filling Edition)

**System Instruction (for the AI that will run this prompt)**
당신은 *“Handover-GPT”*입니다.
아래 지침에 따라 **제공된 컨텍스트 데이터(커밋 메시지, 사용자 추가 정보 등)**를 스스로 분석하고, 사람이 한 줄도 채우지 않아도 **완전한 인수인계 문서**를 생성해야 합니다. (주: 실제 대화 기록, 전체 코드 저장소, 작업 로그, 외부 메타데이터 직접 접근은 현재 이 스크립트의 AI 호출 방식에서는 지원되지 않으며, 요약된 정보가 컨텍스트로 제공됩니다.)

---

## 1. 작업 기본 정보 (Context) — *AI가 아래 제공된 정보를 바탕으로 자동 채우기 시도*
- **작업 이름/ID**: (외부에서 `task`로 주어지는 작업 이름을 사용하거나, 커밋 메시지 등에서 가장 적절한 내용을 추출)
- **작업 수행 기간**: (제공된 Git 커밋 로그의 최초/최종 날짜를 기반으로 자동 탐지 시도) → `YYYY-MM-DD ~ YYYY-MM-DD` 형식
- **담당자**: (제공된 Git 커밋 로그의 author 정보를 기반으로 추정)
- **관련 프로젝트/서비스**: (스크립트 실행 위치의 Git 저장소 이름 또는 사용자가 명시한 프로젝트명으로 추정)

## 2. AI의 역할 및 목표 (Role & Goal)
> “너는 수집한 모든 정보를 요약·정리해 **다음 담당자가 5분 안에 업무 파악**이 가능하도록 Markdown 형식 인수인계 문서를 작성한다.”

- 목표: 아래 섹션 3의 각 항목과 섹션 4, 5의 요구사항에 맞춰 핵심 정보를 누락 없이, 중복 없이, **불릿+서술 혼합** 스타일로 정리한다.

## 3. 작업 상세 내용 (Input Data → AI가 제공된 컨텍스트에서 추출)

**주어진 컨텍스트:**
스크립트로부터 다음 형식의 정보가 제공될 것입니다:
```
### [작업 이름] (사용자가 입력한 작업 이름)

### 최근 Git 활동 요약:
- 날짜: YYYY-MM-DD, 작성자: AuthorName, 제목: Commit Subject (해시: short_hash)
  변경 파일:
    - path/to/file1.py (변경 라인: N, 추가: X, 삭제: Y)
    - path/to/file2.java (변경 라인: M, 추가: A, 삭제: B)
- (다른 커밋 정보들...)

### 사용자 추가 컨텍스트:
(사용자가 입력한 추가적인 설명, 결정 배경, 미팅 요약 등)

### 현재 아티팩트 파일 목록:
(file1.zip, report.pdf 등)
```

**AI 추출 지침:**

### 3.1 주요 목표
- "사용자 추가 컨텍스트" 또는 "최근 Git 활동 요약"의 커밋 메시지에서 **‘목표·Purpose·Goal’** 관련 내용을 찾아 최대 3개 추출·요약. 명시적 언급이 없으면 작업 이름과 커밋 내용으로 추론.

### 3.2 진행 과정 & 변경 사항
- "최근 Git 활동 요약"의 시간순 커밋 로그를 바탕으로 각 날짜별 **주요 액션(커밋 제목)과 변경된 파일(주요 파일 1~2개 명시)**을 불릿으로 정리.
- 커밋 제목에 이슈 번호나 PR 번호가 있다면 함께 언급.

### 3.3 핵심 결정 사항 & 이유
- "사용자 추가 컨텍스트"에서 “결정/결론/choose/채택” 등의 키워드 및 그 이유/배경을 찾아 요약.
- 정보가 부족하면, 주요 커밋(예: Refactor, Design, Add Feature 등)의 내용을 바탕으로 추론하여 기술.

### 3.4 기술 스택 & 환경
- "사용자 추가 컨텍스트"에 명시된 내용이 있다면 우선 사용.
- (AI의 직접적인 파일 스캔은 불가하므로) 만약 커밋된 파일명에 `requirements.txt`, `package.json`, `Dockerfile`, `pom.xml`, `build.gradle` 등이 보이면 해당 파일이 기술 스택 정보를 포함할 수 있다고 언급.
- CI/CD 도구나 OS 정보는 "사용자 추가 컨텍스트"에서 찾아보고, 없으면 "명시된 정보 없음"으로 표기.

### 3.5 데이터 정보 (선택)
- "사용자 추가 컨텍스트" 또는 커밋 메시지/변경된 파일명에서 `migrations`, `schema`, `*.sql` 등의 키워드가 보이면 관련 내용을 요약.
- 없으면 "해당 사항 없음" 또는 "명시된 정보 없음"으로 표기.

### 3.6 결과 & 산출물
- "사용자 추가 컨텍스트"에 명시된 결과 또는 성과를 요약.
- "현재 아티팩트 파일 목록"에 있는 파일들을 나열.
- Git 커밋 로그에서 릴리스 태그나 빌드 관련 메시지가 보이면 언급.

### 3.7 미해결 문제 / 주의 사항
- "사용자 추가 컨텍스트"에서 ‘TODO/FIXME/주의/버그/미해결’ 등의 키워드를 찾아 요약.
- 커밋 메시지에서 관련 내용을 추론.

### 3.8 다음 작업자 제언
- "사용자 추가 컨텍스트"에서 "Tip/주의/참고/연락처" 등의 정보를 찾아 정리.
- 없으면 "특별한 제언 없음"으로 표기.

## 4. 출력 형식
- **문서 형식**: Markdown
- **헤더 구조** (아래 구조를 정확히 따를 것. 대괄호 안은 AI가 채울 내용):
    # [작업 이름/ID] - 인수인계 문서
    ## 1. 작업 개요
    ### 1.1. 작업 기간: [YYYY-MM-DD ~ YYYY-MM-DD]
    ### 1.2. 담당자: [이름]
    ### 1.3. 관련 프로젝트/서비스: [프로젝트명]
    ## 2. 주요 목표
    ## 3. 진행 과정 및 주요 변경 사항
    ## 4. 핵심 결정 사항 및 그 이유
    ## 5. 사용된 기술 스택 및 환경
    ## 6. 데이터 관련 변경 사항 (해당 시)
    ## 7. 결과 및 주요 산출물
    ## 8. 미해결 문제 및 주의 사항
    ## 9. 다음 작업자를 위한 제언
- 각 섹션:
- ‘3. 진행 과정 및 주요 변경 사항’ → **시간순 불릿** (날짜별 그룹화 권장)
- ‘4. 핵심 결정 사항 및 그 이유’ → 서술형 (결정 ▶ 이유 ▶ (있다면) 대안)
- 첫 등장하는 주요 기술 용어는 `( )` 안에 짧은 정의 또는 영문 원어 병기.

## 5. 제약 & 필터
- 언어: **한국어**를 기본으로 사용하되, 기술 용어, 라이브러리명, 파일명 등은 원문(주로 영어)을 그대로 사용하거나 병기.
- 민감 정보(API 키, 비밀번호, 개인 식별 정보 등)는 절대 포함하지 말고, 발견 시 `[민감 정보 마스킹됨]` 또는 `***`로 표기.
- 코드 스니펫은 핵심 로직을 설명하기 위한 최소한의 예시만 허용 (5줄 이내). 긴 코드는 Git 저장소 내 파일 경로로 안내.
- 총 길이는 너무 길지 않게, 핵심 정보 위주로 요약 (예: Markdown 기준 약 150~200줄 내외 목표).

---

### ✅ 실행 단계 (AI가 내부적으로 수행할 프로세스)
1.  **제공된 컨텍스트 데이터**(작업 이름, 최근 Git 활동 요약, 사용자 추가 컨텍스트, 아티팩트 파일 목록)를 면밀히 분석합니다.
2.  위 "3. 작업 상세 내용"의 각 항목별 **AI 추출 지침**에 따라 필요한 정보를 추출하고 정제합니다.
3.  "4. 출력 형식"에 명시된 **헤더 구조**와 스타일 가이드, 그리고 "5. 제약 & 필터"를 **반드시 준수**하여 Markdown 문서를 생성합니다.
"""
).strip()


# --- AI 백엔드 로딩 ---
AIBaseBackend = None
available_backends: Dict[str, Type[AIBaseBackend]] = {}
if BACKEND_DIR.exists() and BACKEND_DIR.is_dir():
    try:
        base_spec = importlib.util.spec_from_file_location("backends.base", BACKEND_DIR / "base.py")
        if base_spec and base_spec.loader:
            backends_base_module = importlib.util.module_from_spec(base_spec)
            sys.modules['backends.base'] = backends_base_module
            base_spec.loader.exec_module(backends_base_module)
            AIBaseBackend = getattr(backends_base_module, 'AIBaseBackend', None)
        else:
            print(f"[yellow]경고: backends.base 모듈 스펙 로딩 실패. AI 기능 사용 불가.[/]")

        if AIBaseBackend:
            for f_py in BACKEND_DIR.glob("*.py"):
                module_name_stem = f_py.stem
                if module_name_stem == "__init__" or module_name_stem == "base":
                    continue
                try:
                    full_module_name = f"backends.{module_name_stem}"
                    spec = importlib.util.spec_from_file_location(full_module_name, f_py)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[full_module_name] = module
                        spec.loader.exec_module(module)
                        for name, obj in module.__dict__.items():
                            if (isinstance(obj, type) and
                                    issubclass(obj, AIBaseBackend) and
                                    obj is not AIBaseBackend):
                                backend_name_from_class = obj.get_name()
                                if backend_name_from_class != "base":
                                    available_backends[backend_name_from_class] = obj
                    else:
                        print(f"[yellow]경고: 백엔드 모듈 스펙 로딩 실패 {f_py.name}[/]")
                except Exception as e_load_backend:
                    print(f"[yellow]경고: 백엔드 파일 '{f_py.name}' 처리 중 예외 발생: {e_load_backend}[/]")
    except Exception as e_base_load:
        print(f"[yellow]경고: backends.base 모듈 처리 중 오류 발생. AI 기능 사용 불가: {e_base_load}[/]")
else:
    print(f"[yellow]경고: 백엔드 디렉토리 '{BACKEND_DIR}'를 찾을 수 없습니다. AI 기능 사용 불가.[/]")


# --- AIProvider 클래스 ---
class AIProvider:
    def __init__(self, backend_name: str, config: Dict[str, Any]):
        self.backend: Optional[AIBaseBackend] = None
        if backend_name == "none": return
        if not AIBaseBackend: return
        if not available_backends:
            print(f"[yellow]경고: 사용 가능한 AI 백엔드가 없어 AIProvider ('{backend_name}') 초기화 불가.[/]")
            return
        if backend_name not in available_backends:
            print(f"[yellow]경고: 알 수 없는 AI 백엔드 '{backend_name}'. AI 기능 비활성화됨. 사용 가능: {list(available_backends.keys())}[/]")
            return
        BackendClass = available_backends[backend_name]
        try:
            self.backend = BackendClass(config)
        except Exception as e:
            print(f"[bold red]오류: AI 백엔드 '{backend_name}' 초기화 실패: {e}[/]")
            if hasattr(BackendClass, 'get_config_description'):
                print(f"[yellow]필요 설정:\n{BackendClass.get_config_description()}[/]")
            self.backend = None

    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        if not self.backend:
            raise RuntimeError("AI 백엔드가 설정되지 않아 요약을 생성할 수 없습니다.")

        # AI 입력은 ⓐ시스템지침(템플릿) ⓑ작업명 ⓒ컨텍스트(데이터) 순으로 합쳐 전달
        # AI 백엔드의 make_summary는 이제 이 merged_ctx를 전체 "사용자 프롬프트"로 받게 됨
        # AI 백엔드의 시스템 프롬프트는 "너는 Handover-GPT이며, 사용자 프롬프트로 전달된 지침 템플릿과 데이터를 기반으로 문서를 작성하라"는 역할만 부여.
        merged_ctx = (
            f"{SYSTEM_INSTRUCTION_MD}\n\n"
            f"### [작업 이름 (Handover 스크립트에서 전달됨)]\n{task}\n\n" # 작업 이름을 명시적으로 전달
            f"--- 제공된 데이터 (아래 내용을 위 템플릿에 맞춰 채워주세요) ---\n\n"
            f"{ctx}" # ctx는 Git 요약 + 사용자 추가 컨텍스트를 포함
        )
        # AI 백엔드의 make_summary에는 task를 별도로 전달하지 않고, merged_ctx에 포함된 작업 이름을 활용하도록 유도
        # 또는, AI 백엔드가 task와 merged_ctx를 별도로 받아 처리하도록 할 수도 있음 (현재는 merged_ctx에 통합)
        return self.backend.make_summary(task, merged_ctx, arts) # task는 여전히 전달 (AI 백엔드에서 활용 여부 결정)

    def verify_summary(self, md: str) -> Tuple[bool, str]:
        if not self.backend: raise RuntimeError("AI 백엔드가 설정되지 않아 요약을 검증할 수 없습니다.")
        try:
            backend_is_ok, backend_msg = self.backend.verify_summary(md)
            print("-" * 20 + " Backend Raw Response " + "-" * 20)
            print(f"[DEBUG] Raw backend_is_ok: {backend_is_ok} (Type: {type(backend_is_ok)})")
            print(f"[DEBUG] Raw backend_msg:\n{backend_msg}")
            print("-" * 50)
        except Exception as e_backend:
            print(f"[DEBUG] Backend verify_summary call failed: {e_backend}")
            return False, f"백엔드 검증 호출 실패: {e_backend}"

        is_ok = backend_is_ok
        msg = backend_msg

        if is_ok:
            print("[DEBUG] Backend reported OK. Starting internal structure checks...")
            lines = md.strip().split('\n')
            headers = [l.strip() for l in lines if l.startswith('#')]
            
            # Self-Filling Prompt 템플릿의 "4. 출력 형식"에 정의된 헤더 구조를 따라야 함
            # 여기서는 AI가 템플릿을 잘 따랐다고 가정하고, 기본적인 검증만 수행
            # (예: 최소한 제목 헤더가 존재하는지)
            if not headers or not headers[0].startswith("# "):
                 is_ok = False
                 msg = "AI 생성 문서에 제목 헤더(#)가 없거나 형식이 잘못되었습니다."
                 print(f"[DEBUG] Internal Check FAILED: {msg}")
            # else:
                # 더 정교한 검증: 템플릿의 헤더 구조와 실제 생성된 헤더를 비교
                # required_header_prefixes = ["# ", "## 1. ", "### 1.1. ", ...] # 템플릿에서 추출 또는 정의
                # ... (상세 검증 로직) ...

            if is_ok:
                print("[DEBUG] Internal structure checks PASSED (or basic check passed).")
        else:
            print("[DEBUG] Backend reported NOT OK. Skipping internal checks.")

        print(f"[DEBUG] AIProvider.verify_summary FINAL return: is_ok={is_ok}, msg='{msg}'")
        return is_ok, msg

    def load_report(self, md: str) -> str:
        if not self.backend: raise RuntimeError("AI 백엔드가 설정되지 않아 보고서를 로드할 수 없습니다.")
        return self.backend.load_report(md)

# --- GitRepo 클래스 ---
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
                    match = re.match(r"(.+?)\s+\|\s+(\d+)\s*([+\-.]*)?", line_stripped)
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
        if not self.repo: return None
        try:
            if self.repo.head.is_detached:
                return f"DETACHED_HEAD@{self.repo.head.commit.hexsha[:7]}"
            return self.repo.active_branch.name
        except Exception as e:
            print(f"[yellow]경고: 현재 브랜치 이름 가져오기 실패: {e}[/yellow]")
            return None

    def get_last_state_commit(self) -> Optional[Commit]:
        if not self.repo: return None
        try:
            for c in self.repo.iter_commits(max_count=200, first_parent=True):
                if c.message.startswith(COMMIT_TAG): return c
        except Exception as e:
            print(f"[yellow]경고: 마지막 상태 커밋 검색 중 오류: {e}[/yellow]")
        return None

    def get_commit_messages_since(self, commit_hash: Optional[str]) -> str:
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
        if not self.repo: raise RuntimeError("Git 저장소가 없어 저장할 수 없습니다.")
        paths_to_add_abs = []
        for p in state_paths:
            paths_to_add_abs.append(str(p.resolve()))
        if snapshot_dir and snapshot_dir.exists() and any(snapshot_dir.iterdir()):
            paths_to_add_abs.append(str(snapshot_dir.resolve()))

        if not paths_to_add_abs: raise ValueError("저장할 상태 파일 또는 스냅샷 파일이 없습니다.")
        try: self.repo.index.add(paths_to_add_abs)
        except Exception as e_add: raise RuntimeError(f"Git add 작업 실패: {e_add}")

        commit_msg = f"{COMMIT_TAG}{task})"
        commit_hash = ""
        try:
            self.repo.index.commit(commit_msg)
            commit_hash = self.repo.head.commit.hexsha[:8]
        except HookExecutionError as e_hook:
            stderr_msg = str(e_hook.stderr) if hasattr(e_hook, 'stderr') else str(e_hook)
            wsl_error_match = re.search(r"execvpe\(/bin/bash\) failed: No such file or directory", stderr_msg)
            detailed_error = f"Git pre-commit 훅 실패 (WSL bash 실행 오류): {stderr_msg}" if wsl_error_match else f"Git pre-commit 훅 실패 (종료 코드: {e_hook.status}): {stderr_msg}"
            raise RuntimeError(detailed_error) from e_hook
        except Exception as e_commit:
            is_nothing_to_commit = False
            if hasattr(e_commit, 'stderr') and isinstance(getattr(e_commit, 'stderr'), str):
                if "nothing to commit" in getattr(e_commit, 'stderr').lower() or \
                   "no changes added to commit" in getattr(e_commit, 'stderr').lower():
                    is_nothing_to_commit = True
            if is_nothing_to_commit or (not self.repo.is_dirty(index=True, working_tree=False) and self.repo.head.commit.message == commit_msg) :
                 print("[yellow]경고: 커밋할 변경 사항이 없습니다. 이전 상태와 동일할 수 있습니다.[/yellow]")
                 commit_hash = self.repo.head.commit.hexsha[:8] + " (변경 없음)"
            else:
                raise RuntimeError(f"Git 커밋 중 오류 발생: {e_commit}") from e_commit

        if self.repo.remotes:
            try:
                current_branch_name = self.get_current_branch_name()
                if current_branch_name and not current_branch_name.startswith("DETACHED_HEAD"):
                    print(f"[dim]'{current_branch_name}' 브랜치를 원격 저장소(origin)에 푸시 시도...[/]")
                    self.repo.git.push('origin', current_branch_name)
                    print("[green]원격 저장소에 푸시 완료.[/]")
                else:
                    print(f"[yellow]경고: 현재 브랜치({current_branch_name})가 특정되지 않았거나 Detached HEAD 상태이므로 푸시를 건너뜁니다.[/]")
            except GitCommandError as e_push:
                print(f"[yellow]경고: 원격 저장소 푸시 실패. 로컬에는 커밋되었습니다. ({e_push.stderr})[/]")
            except Exception as e_general:
                print(f"[yellow]경고: 원격 저장소 푸시 중 예기치 않은 오류: {e_general}[/]")
        else:
            print("[yellow]경고: 설정된 원격 저장소가 없어 푸시를 건너뜁니다.[/]")
        return commit_hash

    def _get_relative_path_str(self, target_path: pathlib.Path) -> Optional[str]:
        if not self.repo_root_path: return None
        try:
            return target_path.resolve().relative_to(self.repo_root_path).as_posix()
        except ValueError: return None
        except Exception: return None

    def list_states(self, current_app_state_dir: pathlib.Path) -> List[Dict]:
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
                "hash": c.hexsha[:8], "task": c.message[len(COMMIT_TAG):-1].strip(), 
                "time": datetime.datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d %H:%M"), "head": headline or "-"
            })
        return list(reversed(items))

    def load_state(self, commit_hash: str, current_app_state_dir: pathlib.Path) -> str:
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
            raise RuntimeError(f"커밋 '{commit_hash}' 상태 로드 중 예기치 않은 오류: {e}")

# --- Serializer 클래스 ---
class Serializer:
    @staticmethod
    def _calculate_sha256(fp: pathlib.Path) -> Optional[str]:
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
    def _generate_html(md: str, title: str) -> str:
        css = """<style>body{font-family:sans-serif;line-height:1.6;padding:20px;max-width:800px;margin:auto;color:#333}h1,h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.5em;margin-bottom:1em}h1{font-size:2em}h2{font-size:1.5em}ul,ol{padding-left:2em}li{margin-bottom:.5em}code{background-color:#f0f0f0;padding:.2em .4em;border-radius:3px;font-family:monospace;font-size:.9em}pre{background-color:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}pre code{background-color:transparent;padding:0;border-radius:0}blockquote{border-left:4px solid #ccc;padding-left:1em;color:#666;margin-left:0}table{border-collapse:collapse;width:100%;margin-bottom:1em}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}</style>"""
        try:
            body_html = markdown2.markdown(md, extras=["metadata","fenced-code-blocks","tables","strike","task_list","code-friendly","markdown-in-html"])
            title_meta = title
            if hasattr(body_html,"metadata") and body_html.metadata.get("title"): title_meta = body_html.metadata["title"]
            return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{textwrap.shorten(title_meta, width=50, placeholder="...")}</title>{css}</head><body>{body_html}</body></html>"""
        except Exception as e:
            print(f"[yellow]경고: Markdown -> HTML 변환 중 오류 발생: {e}[/]")
            escaped_md = "".join(c if c.isalnum() or c in " .,;:!?/\\#$%&'()*+-=<>[]_{}|`~" else f"&#{ord(c)};" for c in md) # type: ignore
            return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>HTML 생성 오류</title></head><body><h1>HTML 생성 오류</h1><p>Markdown 내용을 표시하는 데 문제가 발생했습니다:</p><pre>{escaped_md}</pre></body></html>"""

    @staticmethod
    def save_state(md: str, task: str, current_app_state_dir: pathlib.Path, current_app_art_dir: pathlib.Path, current_app_root: pathlib.Path) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S"); safe_task = "".join(c for c in task if c.isalnum() or c in (' ','_','-')).strip().replace(' ','_');
        if not safe_task: safe_task="untitled_task";
        base_fn = f"{ts}_{safe_task}"
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
            try: html_f.write_text(html_content, encoding="utf-8"); html_ok = True
            except IOError as e: print(f"[yellow]경고: HTML 파일 저장 실패 ({html_f.name}): {e}[/]")
            except Exception as e: print(f"[yellow]경고: HTML 파일 저장 중 예외 ({html_f.name}): {e}[/]")

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

        headline = task
        for ln in md.splitlines():
            ln_s = ln.strip()
            if ln_s.startswith("#"): headline = ln_s.lstrip('# ').strip(); break

        meta = {"task":task,"ts":ts,"headline":headline,"artifact_checksums":checksums}
        try: meta_f.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        except IOError as e: raise RuntimeError(f"메타데이터 파일 저장 실패 ({meta_f}): {e}") from e
        except Exception as e: raise RuntimeError(f"메타데이터 JSON 생성/저장 실패: {e}") from e

        paths_to_commit = [state_f, meta_f]
        if html_ok and html_f.exists(): paths_to_commit.append(html_f)
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
    def multiline(label: str, default: str = "", help_text: Optional[str] = None) -> str:
        UI.console.print(f"\n[bold cyan]{label}[/]")
        if help_text:
            UI.console.print(f"[dim]{help_text}[/dim]")
        
        if default:
            UI.console.print(Panel(default, title="[dim]제공된 내용 (편집 가능, 완료는 빈 줄에서 Enter 두 번)[/dim]", border_style="dim yellow", expand=False, padding=(0,1)))
            UI.console.print("[dim]위 내용을 수정하거나, 그대로 사용하려면 바로 Enter 두 번 입력하세요.[/dim]")
        else:
            UI.console.print("[dim](입력 완료: 빈 줄에서 Enter 두 번)[/dim]")

        lines = []
        blank_count = 0
        initial_input_done = False
        
        while True:
            try: line = input("> " if not initial_input_done and not default else "") 
            except EOFError: break
            
            if line == "" :
                blank_count += 1
            else:
                blank_count = 0
                if not initial_input_done: 
                    initial_input_done = True
                    if default: 
                        lines = [] 
                lines.append(line)

            if blank_count >= 2: break
        
        final_content = "\n".join(lines).strip()

        if not initial_input_done and default:
            print("[dim]입력 없음, 제공된 기본 내용을 사용합니다.[/dim]")
            return default
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
            print(f"[red]{txt}[/]")
            return
        try:
            syntax_obj = Syntax(txt,"diff",theme="dracula",line_numbers=False,word_wrap=False)
            UI.console.print(Panel(syntax_obj,title=f"[bold]Diff (vs {target})[/]",border_style="yellow",expand=True))
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

        try:
            git_repo_candidate = GitRepo(self.app_root)
            if git_repo_candidate.repo:
                self.git = git_repo_candidate
        except Exception as e_git_init:
            self.ui.error(f"GitRepo 객체 생성 실패 (Git 기능 사용 불가): {e_git_init}", traceback.format_exc())

        if backend_choice != "none":
            if available_backends:
                try:
                    self.ai = AIProvider(backend_name=backend_choice, config={})
                    if self.ai and not self.ai.backend:
                        self.ui.error(f"AI 백엔드 '{backend_choice}'가 성공적으로 초기화되지 않았습니다. AI 기능 사용 불가.", None)
                except Exception as e_ai_init:
                     self.ui.error(f"AIProvider ('{backend_choice}') 설정 중 오류 발생. AI 기능 사용 불가.", traceback.format_exc())
            else:
                self.ui.error(f"사용 가능한 AI 백엔드가 없습니다. AI 기능 사용 불가. (선택: {backend_choice})")

    def _ensure_prereqs(self,cmd:str,needs_git:bool,needs_ai:bool):
        if needs_git and (not self.git or not self.git.repo):
            self.ui.error(f"'{cmd}' 명령은 유효한 Git 저장소 내에서 실행해야 합니다."); sys.exit(1)
        if needs_ai:
            if not self.ai or not self.ai.backend:
                self.ui.error(f"'{cmd}' 명령은 AI 백엔드가 성공적으로 설정되어야 합니다. ('--backend' 옵션 및 해당 백엔드 설정을 확인하세요)"); sys.exit(1)

    def save(self):
        self._ensure_prereqs("save", True, True)
        try:
            default_task_name = "작업 요약"
            if self.git and self.git.repo:
                current_branch_name = self.git.get_current_branch_name()
                if current_branch_name and not current_branch_name.startswith("DETACHED_HEAD"):
                    default_task_name = current_branch_name
                elif self.git.repo.head.is_valid() and self.git.repo.head.commit:
                    default_task_name = self.git.repo.head.commit.summary
            task_name_input = self.ui.task_name(default=default_task_name)

            git_commits_info_str = "최근 Git 활동 정보 없음."
            if self.git:
                self.ui.console.print("\n[dim]최근 Git 커밋 정보를 수집 중입니다...[/dim]")
                num_commits_to_fetch = int(os.getenv("HANDOVER_N_COMMITS", 10))
                recent_commits_data = self.git.collect_recent_commits_info(num_commits=num_commits_to_fetch)
                
                if recent_commits_data:
                    formatted_commits = []
                    # AI가 시간순으로 처리하기 좋도록 오래된 커밋부터 전달
                    for commit in reversed(recent_commits_data):
                        commit_line = f"- 날짜: {commit['date']}, 작성자: {commit['author']}, 제목: {commit['subject']} (해시: {commit['hash'][:7]})"
                        changes = commit.get("changes", {})
                        files = changes.get("files", [])
                        if files:
                            commit_line += "\n  변경 파일:"
                            for f_info in files[:3]: # 예시로 최대 3개 파일 정보만
                                commit_line += f"\n    - {f_info['file']} (변경 라인: {f_info.get('changed_lines', 'N/A')}, 추가: {f_info.get('insertions', 'N/A')}, 삭제: {f_info.get('deletions', 'N/A')})"
                            if len(files) > 3:
                                commit_line += f"\n    - ... (외 {len(files) - 3}개 파일 변경)"
                        formatted_commits.append(commit_line)
                    git_commits_info_str = "\n".join(formatted_commits)
                    self.ui.panel(git_commits_info_str, "수집된 Git 커밋 요약 (AI 전달용)", border_style="green")
                else:
                    self.ui.console.print("[yellow]Git 커밋 정보를 수집하지 못했거나, 최근 커밋이 없습니다.[/yellow]")
            
            user_additional_context = self.ui.multiline(
                "추가 컨텍스트 또는 강조 사항 (선택 사항, AI가 템플릿의 다른 부분을 채우는 데 사용)",
                help_text="AI가 Git 정보 외에 참고할 내용을 입력하세요. (예: 주요 결정 배경, 미팅 요약, Self-Filling Prompt의 다른 섹션 내용 등)"
            )
            
            # AI에게 전달할 전체 컨텍스트: 시스템 지침(템플릿) + 작업명 + 데이터(Git 요약 + 사용자 추가 컨텍스트)
            # AIProvider.make_summary 내부에서 SYSTEM_INSTRUCTION_MD와 task, ctx를 합쳐서 최종 프롬프트 구성
            # ctx 부분만 데이터로 구성
            data_context_for_ai = "### 최근 Git 활동 요약:\n"
            data_context_for_ai += git_commits_info_str
            if user_additional_context:
                data_context_for_ai += "\n\n### 사용자 추가 컨텍스트:\n" + user_additional_context
            
            current_artifact_files = []
            if self.art_dir.exists() and self.art_dir.is_dir():
                current_artifact_files = [f.name for f in self.art_dir.iterdir() if f.is_file()]

            self.ui.console.print("\n[bold yellow]AI가 인수인계 문서 초안을 생성 중입니다... (Self-Filling Prompt 및 데이터 기반)[/bold yellow]")
            
            # AIProvider.make_summary는 이제 task와 data_context_for_ai를 받음
            # 내부적으로 SYSTEM_INSTRUCTION_MD와 결합됨
            generated_markdown_draft = self.ai.make_summary(
                task=task_name_input, 
                ctx=data_context_for_ai, 
                arts=current_artifact_files
            )
            self.ui.panel(generated_markdown_draft, "AI 생성 인수인계 문서 초안")

            self.ui.console.print("\n[bold green]AI가 생성한 초안입니다. 내용을 검토하고 최종적으로 수정/완성해주세요.[/bold green]")
            final_markdown_content = self.ui.multiline(
                "최종 인수인계 문서 내용 편집",
                default=generated_markdown_draft,
                help_text="위 초안을 바탕으로 최종 문서를 완성하세요."
            )

            if not final_markdown_content.strip():
                self.ui.error("인수인계 문서 내용이 비어있어 저장을 취소합니다."); return

            self.ui.console.print("[bold yellow]수정된 내용을 AI가 다시 검증 중입니다...[/bold yellow]")
            is_valid_summary, validation_message = self.ai.verify_summary(final_markdown_content)
            if not is_valid_summary:
                raise RuntimeError(f"사용자가 수정한 인수인계 문서 검증 실패:\n{validation_message}")
            else:
                self.ui.notify("AI 최종 검증 통과!", style="green")

            saved_state_files, artifact_snapshot_dir = Serializer.save_state(final_markdown_content, task_name_input, self.state_dir, self.art_dir, self.app_root)
            
            if not self.git:
                 self.ui.error("Git 저장소가 설정되지 않아 상태를 커밋할 수 없습니다."); return
            commit_short_hash = self.git.save(saved_state_files, task_name_input, artifact_snapshot_dir)
            self.ui.notify(f"인수인계 상태 저장 완료! (Commit: {commit_short_hash})", style="bold green")

            generated_html_file = next((f for f in saved_state_files if f.name.endswith(".html")), None)
            if generated_html_file and generated_html_file.exists():
                self.ui.console.print(f"[dim]HTML 프리뷰 생성됨: {generated_html_file.relative_to(self.app_root)}[/dim]")

        except Exception as e: self.ui.error(f"Save 작업 중 오류 발생: {str(e)}", traceback.format_exc())


    def load(self, latest: bool = False):
        self._ensure_prereqs("load", True, True) 
        try:
            if not self.git:
                self.ui.error("Git 저장소가 설정되지 않아 상태를 로드할 수 없습니다."); return
            saved_states_list = self.git.list_states(self.state_dir)
            if not saved_states_list: self.ui.error("저장된 인수인계 상태가 없습니다."); return
            selected_commit_hash: Optional[str]
            if latest:
                if not saved_states_list:
                    self.ui.error("저장된 상태가 없어 최근 상태를 로드할 수 없습니다."); return
                selected_commit_hash = saved_states_list[-1]["hash"]
                print(f"[info]가장 최근 상태 로드 중: {selected_commit_hash} (작업명: {saved_states_list[-1]['task']})[/]")
            else: selected_commit_hash = self.ui.pick_state(saved_states_list)
            if not selected_commit_hash: return
            self.ui.console.print(f"[bold yellow]{selected_commit_hash} 커밋에서 상태 정보를 로드 중입니다...[/]")
            markdown_content = self.git.load_state(selected_commit_hash, self.state_dir)
            self.ui.panel(markdown_content, f"로드된 인수인계 문서 (Commit: {selected_commit_hash})", border_style="cyan")
            if self.ai and self.ai.backend:
                self.ui.console.print("[bold yellow]AI가 로드된 상태를 분석하고 이해도를 보고합니다...[/bold yellow]")
                ai_report = self.ai.load_report(markdown_content)
                self.ui.panel(ai_report, "AI 이해도 보고서", border_style="magenta")
        except Exception as e: self.ui.error(f"Load 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def diff(self, target: str = "HEAD"):
        self._ensure_prereqs("diff", True, False)
        try:
            if not self.git:
                self.ui.error("Git 저장소가 설정되지 않아 diff를 생성할 수 없습니다."); return
            self.ui.console.print(f"[bold yellow]'{target}' 대비 현재 변경 사항을 확인 중입니다... (Git 추적 파일 기준)[/]")
            diff_output_text = self.git.get_diff(target, color=True)
            self.ui.diff_panel(diff_output_text, target)
        except Exception as e: self.ui.error(f"Diff 작업 중 오류 발생: {str(e)}", traceback.format_exc())

    def verify_checksums(self, commit_hash: str):
        self._ensure_prereqs("verify", True, False)
        if not self.git or not self.git.repo :
            self.ui.error("Git 저장소가 설정되지 않아 체크섬을 확인할 수 없습니다."); return
        self.ui.console.print(f"[dim]커밋 {commit_hash}의 저장된 아티팩트 체크섬 정보를 표시합니다.[/]")
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
        except Exception as e: self.ui.error(f"체크섬 정보 로드/표시 중 오류 ({commit_hash}): {str(e)}", traceback.format_exc())


# --- 스크립트 진입점 ---
def main_cli_entry_point():
    cli_root_path = pathlib.Path('.').resolve()
    is_git_repo_at_cli_root = False
    try:
        git_repo_candidate = Repo(str(cli_root_path), search_parent_directories=True)
        found_root_path = pathlib.Path(git_repo_candidate.working_tree_dir)
        cli_root_path = found_root_path
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

    parser = argparse.ArgumentParser(description="AI 기반 프로젝트 인수인계 상태 관리 도구 (v1.1.12 - Self-Filling Prompt)", formatter_class=argparse.RawTextHelpFormatter)
    
    backend_choices_list = list(available_backends.keys()) if available_backends else []
    default_be = "none" 
    env_ai_backend = os.getenv("AI_BACKEND")
    if env_ai_backend and env_ai_backend in backend_choices_list:
        default_be = env_ai_backend
    elif "ollama" in backend_choices_list: 
        default_be = "ollama"
    elif backend_choices_list: 
        default_be = backend_choices_list[0]

    parser.add_argument("--backend", default=default_be,
                        choices=backend_choices_list + ["none"],
                        help=f"AI 백엔드 (기본값: 환경변수 AI_BACKEND 또는 '{default_be}'). 사용가능: {', '.join(backend_choices_list) or '없음'}. 'none'으로 AI 비활성화 시 AI 필요 명령 사용 불가.")

    subparsers = parser.add_subparsers(dest="command", help="실행할 작업", required=True)
    
    cmd_configs = [("save", "최근 Git 변경사항 기반 AI 초안 생성 후 저장", True, True),
                   ("load", "과거 저장된 상태 불러오기 (AI 보고서 가능)", True, True),
                   ("diff", "현재 변경 사항 미리보기", True, False),
                   ("verify", "저장된 상태 아티팩트 체크섬 표시", True, False)]

    for name, help_txt, git_req, ai_req_flag in cmd_configs:
        help_suffix = ""
        if git_req: help_suffix += " (Git 필요)"
        if ai_req_flag: help_suffix += " (AI 백엔드 설정 필요)"
        p = subparsers.add_parser(name, help=f"{help_txt}{help_suffix}")
        if name == "load": p.add_argument("-l", "--latest", action="store_true", help="가장 최근 상태 자동 로드")
        if name == "diff": p.add_argument("target", nargs="?", default="HEAD", help="비교 대상 Commit/Branch (기본값: HEAD)")
        if name == "verify": p.add_argument("commit", help="체크섬 정보를 확인할 상태 커밋 해시")

    args = parser.parse_args()

    chosen_cmd_config = next((c for c in cmd_configs if c[0] == args.command), None)
    if not chosen_cmd_config:
        UI.error(f"알 수 없는 명령어: {args.command}"); sys.exit(1)
    _, _, git_needed, ai_needed_for_cmd = chosen_cmd_config

    if git_needed and not is_git_repo_at_cli_root:
        UI.error(f"'{args.command}' 명령은 Git 저장소 내에서 실행해야 합니다. (현재 위치는 Git 저장소가 아님: {cli_root_path})"); sys.exit(1)

    if ai_needed_for_cmd:
        if args.backend == "none":
            UI.error(f"'{args.command}' 명령 실행 불가: AI 기능이 필요하지만 '--backend' 옵션이 'none'으로 설정되었습니다."); sys.exit(1)
        elif args.backend not in available_backends and args.backend != "none":
            UI.error(f"'{args.command}' 명령 실행 불가: 선택된 AI 백엔드 '{args.backend}'를 로드할 수 없거나 초기화에 실패했습니다."); sys.exit(1)

    print(f"[bold underline]Handover 스크립트 v1.1.12 (Self-Filling Prompt)[/]")
    if is_git_repo_at_cli_root: print(f"[dim]프로젝트 루트 (Git): {cli_root_path}[/dim]")
    else: print(f"[dim]현재 작업 폴더 (Git 저장소 아님): {cli_root_path}[/dim]")

    if args.backend != "none" and args.backend in available_backends:
        print(f"[dim]선택된 AI 백엔드: [bold cyan]{args.backend}[/][/dim]")
    elif args.backend == "none" and not ai_needed_for_cmd:
        print(f"[dim]AI 백엔드: [bold yellow]none (비활성화됨, 현재 명령은 AI 불필요)[/][/dim]")

    try:
        handler = Handover(backend_choice=args.backend, current_app_root=cli_root_path)
        if args.command == "save": handler.save()
        elif args.command == "load": handler.load(latest=args.latest)
        elif args.command == "diff": handler.diff(target=args.target)
        elif args.command == "verify": handler.verify_checksums(commit_hash=args.commit)
    except Exception as e_handler:
        UI.error(f"핸들러 실행 중 예기치 않은 오류: {str(e_handler)}", traceback.format_exc()); sys.exit(1)


if __name__ == "__main__":
    if sys.version_info < (3, 8): print("[bold red]오류: Python 3.8 이상 필요.[/]"); sys.exit(1)
    main_cli_entry_point()
