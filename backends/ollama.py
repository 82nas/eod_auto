#!/usr/bin/env python3
# backends/ollama.py – Ollama AI Backend (v1.3.0 - Self-Filling Prompt 적용)

import os
import re
import textwrap
import requests # requests 라이브러리가 필요합니다.
from typing import List, Tuple, Dict, Any

# AIBaseBackend를 임포트하기 위해 상대 경로 사용
# 이 파일이 handover.py와 같은 폴더 내의 backends 폴더에 있다고 가정합니다.
try:
    from .base import AIBaseBackend
except ImportError:
    # 만약 위 임포트가 실패하면 (예: 직접 실행 시),
    # Python 경로에 맞게 수정하거나, AIBaseBackend의 위치를 명시해야 합니다.
    # 이 예제에서는 일단 플레이스홀더 클래스를 정의합니다.
    class AIBaseBackend:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
        @staticmethod
        def get_name() -> str: return "base"
        @staticmethod
        def get_config_description() -> str: return "Base backend config"
        def make_summary(self, task: str, ctx: str, arts: List[str]) -> str: return "Not implemented"
        def verify_summary(self, md: str) -> Tuple[bool, str]: return False, "Not implemented"
        def load_report(self, md: str) -> str: return "Not implemented"


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
            # 서버 연결 테스트 (timeout 5초)
            response = requests.head(self.base_url, timeout=5)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        except requests.exceptions.RequestException as e:
            # 연결 실패 또는 HTTP 오류 시 ConnectionError 발생
            raise ConnectionError(
                f"Ollama 서버({self.base_url})에 연결할 수 없습니다: {e}"
            ) from e

    # -----------------------------------------------------------------------#
    # 0-2. 메타 정보
    # -----------------------------------------------------------------------#
    @staticmethod
    def get_name() -> str:
        """백엔드의 고유 이름을 반환합니다."""
        return "ollama"

    @staticmethod
    def get_config_description() -> str:
        """백엔드 설정에 대한 설명을 반환합니다."""
        return textwrap.dedent(
            """
            Ollama 설정:
            - model: (선택) 사용할 Ollama 모델 이름 (환경변수: AI_MODEL, 기본값: 'llama3')
            - ollama_base_url: (선택) Ollama API 서버의 기본 URL (환경변수: OLLAMA_BASE_URL, 기본값: 'http://localhost:11434')
            """
        )

    # -----------------------------------------------------------------------#
    # 1. 내부 API 요청 래퍼
    # -----------------------------------------------------------------------#
    def _req(self, sys_msg: str, user_msg: str) -> str:
        """
        Ollama chat API에 요청을 보내고 응답 텍스트를 반환하는 내부 헬퍼 메서드.
        """
        api_url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}, # 사용자 메시지에는 이제 템플릿+데이터가 포함됨
            ],
            "options": {
                "temperature": 0, # 일관성 있는 결과를 위해 온도 0 설정
                "num_ctx": self.config.get("ollama_num_ctx", 4096), # 컨텍스트 길이 설정 (설정 가능하도록)
            },
            "stream": False, # 스트리밍 응답 비활성화
        }
        try:
            response = requests.post(api_url, json=payload, timeout=self.config.get("ollama_timeout", 180)) # timeout 설정 가능하도록
            response.raise_for_status()

            response_data = response.json()
            message_content = (response_data.get("message") or {}).get("content", "")

            if not message_content:
                 print(f"[WARN] Ollama model '{self.model}' returned an empty response for user_msg starting with: {user_msg[:100]}...")
                 return ""

            return message_content.strip()

        except requests.exceptions.Timeout:
             raise RuntimeError(f"Ollama API 호출 시간 초과 ({api_url})")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API 호출 실패 ({api_url}): {e}") from e
        except json.JSONDecodeError:
             raise RuntimeError(f"Ollama API 응답 JSON 파싱 실패 ({api_url}). 응답 내용: {response.text[:500]}...") # type: ignore
        except Exception as e:
            raise RuntimeError(f"Ollama API 요청 중 예기치 않은 오류 발생: {e}") from e

    # -----------------------------------------------------------------------#
    # 2. 인수인계 문서 생성 (Self-Filling Prompt 적용)
    # -----------------------------------------------------------------------#
    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        """
        handover.py로부터 전달받은 Self-Filling Prompt 템플릿과 데이터를 기반으로
        AI가 인수인계 문서를 생성하도록 요청합니다.

        Args:
            task: 작업 이름 (handover.py에서 사용자 입력 또는 Git 정보로 생성).
                  이 정보는 ctx 내의 템플릿에서 "[작업 이름 (Handover 스크립트에서 전달됨)]" 부분에 이미 포함되어 있습니다.
            ctx: handover.py에서 전달된 "merged_ctx"입니다.
                 이 안에는 SELF_FILLING_PROMPT_TEMPLATE (상세 지침)과
                 "--- 제공된 데이터 ---" (Git 요약, 사용자 추가 정보 등)가 포함되어 있습니다.
            arts: 현재 작업 산출물 파일 이름 목록.
        Returns:
            AI가 생성한 Markdown 인수인계 문서 문자열.
        """

        # AI에게 전달할 시스템 메시지: AI의 역할과 입력(ctx)의 성격을 명시합니다.
        # 이 시스템 메시지는 AI가 사용자 메시지(ctx)로 전달될 Self-Filling Prompt 템플릿과 데이터를
        # 올바르게 해석하고 처리하도록 안내하는 핵심적인 역할을 합니다.
        sys_msg = textwrap.dedent(f"""
            당신은 고도로 지능적이고 유능한 "Handover-GPT"입니다.
            이제 사용자 메시지(User Message)를 통해 인수인계 문서 작성을 위한 매우 상세한 지침 템플릿과, 해당 템플릿을 채우기 위한 관련 데이터가 함께 전달될 것입니다.

            **당신의 핵심 임무는 다음과 같습니다:**

            1.  사용자 메시지로 전달된 내용 전체를 주의 깊게 읽고, "# ✨ AI Handover-Prompt (Self-Filling Edition)"로 시작하는 **지침 템플릿 부분**과 그 이후에 "**--- 제공된 데이터 ---**" 섹션으로 구분되어 제공되는 **데이터 부분**을 명확히 인지합니다.
            2.  지침 템플릿에 명시된 **"System Instruction (for the AI that will run this prompt)"**, 각 섹션별 **"자동 채우기" 규칙**, **"AI가 추출" 규칙**, **"4. 출력 형식"**(특히 헤더 구조), 그리고 **"5. 제약 & 필터"** 등을 **반드시 정확하게 준수**해야 합니다.
            3.  지침 템플릿의 상세 규칙에 따라, 함께 제공된 **"--- 제공된 데이터 ---" 부분**(예: 작업 이름 '{task}', 최근 Git 활동 요약, 사용자 추가 컨텍스트 등)과 템플릿에서 언급된 다른 정보 소스(예: 코드 저장소, 커밋 메시지 - 이미 요약본으로 일부 제공됨)를 **스스로 분석하고 종합**하여 지침 템플릿의 각 항목을 채워 넣습니다.
                * 템플릿의 "1. 작업 기본 정보 (Context)" 섹션의 "작업 이름/ID"는 사용자 메시지 내 "### [작업 이름 (Handover 스크립트에서 전달됨)]" 부분에 명시된 '{task}' 값을 사용하십시오.
            4.  최종적으로, 지침 템플릿의 "4. 출력 형식"에 명시된 Markdown 헤더 구조와 스타일 가이드에 맞춰 **완전하고 상세한 인수인계 문서**를 생성해야 합니다. 사람이 추가적인 수정을 거의 하지 않아도 될 만큼 완성도 높은 결과물을 목표로 합니다.
            5.  인수인계 문서의 "산출물" 섹션에는 다음 파일 목록을 참고하여 작성하십시오: {', '.join(arts) if arts else '명시된 산출물 없음'}

            주어진 지침 템플릿과 데이터를 최대한 활용하여, 다음 담당자가 혼란 없이 빠르게 업무를 파악하고 작업을 이어받을 수 있도록 명확하고, 간결하며, 실질적으로 유용한 인수인계 문서를 작성해주십시오.
            **오직 최종 Markdown 인수인계 문서 내용만 응답으로 반환해야 하며, 다른 부가적인 설명이나 이 지침 자체를 응답에 반복해서는 안 됩니다.**
            """)

        # 사용자 메시지는 handover.py에서 이미 SYSTEM_INSTRUCTION_MD (템플릿)와
        # 작업명, 데이터(Git 요약 + 사용자 추가 정보)를 결합하여 전달된 ctx 전체입니다.
        user_msg = ctx

        # Ollama API 호출
        return self._req(sys_msg, user_msg)

    # -----------------------------------------------------------------------#
    # 3. 생성 문서 검증 (PASS / FAIL / OK 모두 허용)
    # -----------------------------------------------------------------------#
    def verify_summary(self, md: str) -> Tuple[bool, str]:
        """
        AI가 생성한 Markdown 인수인계 문서가 기본적인 규칙을 준수하는지 검증합니다.
        (handover.py의 AIProvider.verify_summary에서 더 상세한 내부 구조 검증 가능)
        """
        sys_msg = (
            "당신은 Markdown 형식 검사기입니다. 전달받은 Markdown 문서가 일반적인 인수인계 문서로서 최소한의 구조(예: 제목 헤더 시작)를 갖추고 있는지, "
            "그리고 심각한 형식 위반(예: 내용 없음, 지침 반복 등)이 없는지 확인해주세요. "
            "문제가 없다면 'Validation: PASS' 또는 간단히 'OK'라고만 응답하고, "
            "문제가 있다면 'Validation: FAIL - [구체적인 문제점 요약]' 형식으로 응답하세요."
        )
        user_msg = f"다음 Markdown 문서를 검증하십시오:\n\n```markdown\n{md}\n```"
        res = self._req(sys_msg, user_msg)

        if not res:
            return False, "AI 검증 응답이 비어있습니다."

        res_upper_stripped = res.strip().upper()

        # 성공 패턴: "VALIDATION: PASS" 또는 "OK" (선택적으로 ! 또는 . 포함)
        if res_upper_stripped.startswith("VALIDATION: PASS") or re.fullmatch(r"OK[!.]?", res_upper_stripped):
            # AI 백엔드가 PASS라고 했으므로, handover.py의 내부 검증에서 더 자세히 볼 수 있음
            return True, "AI validation passed (by backend)." # 백엔드 통과 메시지 명시

        # 실패 패턴: "VALIDATION: FAIL - [사유]"
        if res_upper_stripped.startswith("VALIDATION: FAIL"):
            reason = res.strip()[len("Validation: FAIL"):].lstrip(" -:").strip()
            return False, reason if reason else "AI 검증 실패 (백엔드 제공 사유 없음)"

        # 그 외: AI가 PASS/FAIL 형식을 따르지 않았지만, 내용상 문제가 없어 보일 수도 있음.
        # 하지만 일관성을 위해 여기서는 실패로 간주하고 AI의 응답을 그대로 전달.
        # (handover.py의 AIProvider.verify_summary에서 최종 판단)
        # 또는, 여기서 더 관대한 규칙을 적용할 수도 있음 (예: 특정 부정 키워드가 없으면 True)
        # 현재는 명시적인 PASS/FAIL 형식을 기대.
        # 만약 AI가 단순히 "문제 없음" 등으로 응답하는 경우가 잦다면, 아래 로직 수정 필요.
        # print(f"[DEBUG] Ollama verify_summary unexpected response: {res}")
        return False, f"AI 검증 응답이 예상된 형식(PASS/FAIL)이 아님: {res}"


    # -----------------------------------------------------------------------#
    # 4. 보고서 요약
    # -----------------------------------------------------------------------#
    def load_report(self, md: str) -> str:
        """
        과거 저장된 인수인계 문서(Markdown)를 AI가 이해하고 요약 보고서를 생성합니다.
        """
        sys_msg = (
            "당신은 프로젝트 관리자입니다. 전달받은 Markdown 형식의 프로젝트 인수인계 보고서를 분석하고, "
            "현재 상황과 다음 단계를 명확하게 요약해야 합니다."
        )
        user_msg = textwrap.dedent(
            f"""
            다음은 이전 작업자가 작성한 인수인계 보고서입니다.

            ### 인수인계 보고서 내용
            ```markdown
            {md}
            ```
            ### 요청사항
            위 보고서 내용을 바탕으로 다음 두 가지 항목을 간결하게 요약하여 보고해 주십시오:
            1.  **현재 프로젝트 상황 요약:** 1~2 문장으로 핵심 상황을 설명합니다.
            2.  **즉시 수행해야 할 다음 작업:** 가장 중요하고 시급한 다음 할 일 1~3가지를 명확하게 제시합니다.

            요약 보고서만 응답으로 반환하십시오.
            """
        )
        return self._req(sys_msg, user_msg)

