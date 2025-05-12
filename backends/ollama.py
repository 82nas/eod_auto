#!/usr/bin/env python3
# backends/ollama.py – Ollama AI Backend (v1.2.1)

import os
import re
import textwrap
import requests
from typing import List, Tuple, Dict, Any
from .base import AIBaseBackend


# ---------------------------------------------------------------------------#
# 0.  OllamaBackend 클래스
# ---------------------------------------------------------------------------#
class OllamaBackend(AIBaseBackend):
    """Ollama와 상호작용하는 AI 백엔드 구현체."""

    # -----------------------------------------------------------------------#
    # 0-1. 초기화
    # -----------------------------------------------------------------------#
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", os.getenv("AI_MODEL", "llama3"))
        self.base_url = config.get(
            "ollama_base_url",
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        try:
            requests.head(self.base_url, timeout=5).raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Ollama 서버({self.base_url})에 연결할 수 없습니다: {e}"
            ) from e

    # -----------------------------------------------------------------------#
    # 0-2. 메타
    # -----------------------------------------------------------------------#
    @staticmethod
    def get_name() -> str:
        return "ollama"

    @staticmethod
    def get_config_description() -> str:
        return textwrap.dedent(
            """
            Ollama 설정:
            - model: (선택) 사용할 모델 (환경변수 AI_MODEL, 기본 'llama3')
            - ollama_base_url: (선택) Ollama API URL (환경변수 OLLAMA_BASE_URL)
            """
        )

    # -----------------------------------------------------------------------#
    # 1. 내부 API 래퍼
    # -----------------------------------------------------------------------#
    def _req(self, sys_msg: str, user_msg: str) -> str:
        api_url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            "options": {"temperature": 0, "num_ctx": 4096},
            "stream": False,
        }
        try:
            r = requests.post(api_url, json=payload, timeout=180)
            r.raise_for_status()
            content = (r.json().get("message") or {}).get("content", "")
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"Ollama API 호출 실패: {e}") from e

    # -----------------------------------------------------------------------#
    # 2. 인수인계 문서 생성
    # -----------------------------------------------------------------------#
    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        guidelines = textwrap.dedent(
            f"""
            ### 절대 규칙
            1. 헤더 구조(정확히 7개, 순서 고정)
               # {task}
               ## 목표
               ## 진행
               ## 결정
               ## 결과
               ## 다음할일
               ## 산출물
            2. `## 목표`‧`## 진행`‧`## 결정`‧`## 결과`‧`## 다음할일`
               → 불릿(`- `) 2–5개, 한국어 문장, 마침표로 끝남
            3. `## 산출물`
               → 파일명 쉼표 구분. 없으면 **없음** 단어 하나만
            4. 금지: 표, 코드블록, 인라인코드, HTML, 이미지, 체크박스
            5. 설명은 한국어(Oracle·PostgreSQL 등 고유명사 예외)
            6. 지침을 결과 문서에 포함하지 말 것
            """
        )

        sys_msg = (
            "당신은 시니어 개발자입니다. 아래 절대 규칙을 100% 지켜 "
            "Markdown 인수인계 문서를 작성하세요.\n\n" + guidelines
        )

        arts_str = ", ".join(arts) if arts else "없음"
        user_msg = textwrap.dedent(
            f"""
            작업 이름: {task}

            최근 작업 요약:
            {ctx or '제공된 내용 없음'}

            현재 작업 산출물: {arts_str}

            위 정보를 반영하여 절대 규칙을 지킨 Markdown 문서만 반환하십시오.
            """
        )
        return self._req(sys_msg, user_msg)

    # -----------------------------------------------------------------------#
    # 3. 생성 문서 검증 (PASS / FAIL / OK 모두 허용)
    # -----------------------------------------------------------------------#
    def verify_summary(self, md: str) -> Tuple[bool, str]:
        sys_msg = (
            "당신은 Markdown 검사기입니다. 규칙 위반이 있으면 나열하고, "
            "완벽하면 'Validation: PASS'만 반환하거나 짧게 'OK'라고만 답하세요."
        )
        user_msg = f"검증 대상 Markdown:\n```markdown\n{md}\n```"
        res = self._req(sys_msg, user_msg)
        if not res:
            return False, "AI 검증 응답 없음"

        up = res.strip().upper()

        # ① PASS 패턴 --- 'Validation: PASS' 또는 단순 OK(OK, OK!, OK.)
        if up.startswith("VALIDATION: PASS") or re.fullmatch(r"OK[!.]?", up):
            return True, "AI validation passed."

        # ② FAIL 패턴
        if up.startswith("VALIDATION: FAIL"):
            reason = res[len("Validation: FAIL") :].lstrip(" -")
            return False, reason or "AI 검증 실패 (사유 미제공)"

        # ③ 예기치 않은 형식
        return False, f"AI 검증 응답 형식이 예상과 다름: {res}"

    # -----------------------------------------------------------------------#
    # 4. 보고서 요약
    # -----------------------------------------------------------------------#
    def load_report(self, md: str) -> str:
        sys_msg = (
            "당신은 프로젝트 관리자입니다. 전달받은 인수인계 보고서를 읽고 "
            "현재 상황과 즉시 해야 할 일을 간결히 요약하세요."
        )
        user_msg = textwrap.dedent(
            f"""
            ### 보고서
            ```markdown
            {md}
            ```

            1) 현재 상황 요약 (1–2문장)
            2) 즉시 해야 할 일 1–3가지
            """
        )
        return self._req(sys_msg, user_msg)
