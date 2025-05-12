# backends/ollama.py
import os
import re
import requests
import textwrap
import json
from typing import List, Tuple, Dict, Any
from .base import AIBaseBackend

class OllamaBackend(AIBaseBackend):
    """AI Backend implementation for Ollama."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", os.getenv("AI_MODEL", "llama3"))
        self.base_url = config.get("ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        try:
            response = requests.head(self.base_url, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.ConnectionError):
                raise ConnectionError(f"Ollama 서버({self.base_url}) 연결 실패. Ollama가 실행 중인지, URL이 정확한지 확인하세요.") from e
            else:
                raise ConnectionError(f"Ollama 서버({self.base_url}) 확인 중 오류 발생: {e}") from e

    @staticmethod
    def get_name() -> str:
        return "ollama"

    @staticmethod
    def get_config_description() -> str:
        return textwrap.dedent("""
            Ollama 설정:
            - model: (선택) Ollama 모델 이름 (기본값: AI_MODEL 환경변수 또는 'llama3').
            - ollama_base_url: (선택) Ollama 서버 URL (기본값: OLLAMA_BASE_URL 환경변수 또는 'http://localhost:11434').
        """)

    def _req(self, sys_msg: str, user_msg: str) -> str:
        """Sends request to Ollama API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    "options": {"temperature": 0, "num_ctx": 4096},
                    "stream": False
                },
                timeout=180
            )
            response.raise_for_status()

            response_data = response.json()
            message_data = response_data.get("message")
            if isinstance(message_data, dict):
                 message_content = message_data.get("content", "")
            else:
                 print(f"Warning: Unexpected 'message' structure in Ollama response: {message_data}")
                 message_content = ""

            return message_content.strip()

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Ollama API 연결 실패 ({self.base_url}): {e}") from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"Ollama API 응답 시간 초과 (모델: {self.model}): {e}") from e
        except requests.exceptions.RequestException as e:
            err_msg = f"Ollama API 요청 실패 (모델: {self.model}): {e}"
            if e.response is not None:
                try:
                    err_details = e.response.json()
                    err_msg += f"\n상태 코드: {e.response.status_code}\n오류: {err_details.get('error', e.response.text[:500])}"
                except json.JSONDecodeError:
                     err_msg += f"\n상태 코드: {e.response.status_code}\n응답 (텍스트): {e.response.text[:500]}"
            raise RuntimeError(err_msg) from e
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
             response_text_snippet = "응답 내용 확인 불가"
             if 'response' in locals() and hasattr(response, 'text'):
                 response_text_snippet = response.text[:1000]
             raise RuntimeError(f"Ollama API 응답 처리 오류 (모델: {self.model}, 예상치 못한 형식): {e}\n받은 응답 미리보기: {response_text_snippet}") from e

    # --- Start: Modified make_summary with System/User prompt separation ---
    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        # 1) 모든 지침·규칙 블록 → sys_msg (LLM이 출력할 필요 없음)
        sys_msg = textwrap.dedent(f"""
        당신은 한국어로 인수인계 Markdown 파일을 작성하는 시니어 개발자입니다.
        다음 규칙을 **절대 어기지 마십시오**.

        ● 헤더 구조 (정확히 7개):
            # {task}
            ## 목표 이름
            ## 진행
            ## 결정
            ## 결과
            ## 다음할일
            ## 산출물

        ● 각 ## 섹션(산출물 제외) → bullet 2~5개 ( - 로 시작, 마침표로 끝 )
        ● ## 산출물 → 파일이름, 로 리스트 - 없으면 '없음' 단어 하나만
        ● 금지: 표·코드블록·인라인코드·HTML·이미지·체크박스·### 헤더
        ● 모든 서술은 한국어(고유명사 제외)
        ● 응답은 오직 `# {task}` 로 시작하는 7개 헤더의 Markdown 문서 내용만 포함해야 합니다. 다른 설명이나 서론, 지침 반복 등은 절대 포함하지 마십시오.
        """) # 금지 항목에 '### 헤더' 추가 및 최종 출력 형식 강조

        # 2) user_msg 는 실제 데이터만 전달
        arts_str = ", ".join(arts) if arts else "없음" # Correctly handle empty arts list
        user_msg = textwrap.dedent(f"""
        # 작업 이름
        {task}

        # 최근 작업 요약
        {ctx or '제공된 내용 없음'}

        # 현재 산출물 목록
        {arts_str}

        위 정보를 바탕으로, 앞서 system 메시지에서 설명한 규칙에 따라 7개 헤더 구조를 가진 인수인계 Markdown 문서만 생성하여 반환하십시오.
        """) # User 메시지 구조화 및 최종 요구 명확화

        return self._req(sys_msg, user_msg)
    # --- End: Modified make_summary ---

    def verify_summary(self, md: str) -> Tuple[bool, str]:
        # System/User 메시지 구조는 이전과 동일하게 유지 (검증 규칙 전달)
        sys_msg = textwrap.dedent("""
        당신은 Markdown 문서 형식을 검증하는 매우 꼼꼼한 검토자입니다.
        주어진 Markdown 텍스트가 아래 규칙을 모두 준수하는지 확인하고 결과를 보고합니다.
        """)
        user_msg = textwrap.dedent(f"""
        ### 검증 규칙
        1.  **필수 헤더 및 순서:** 다음 7개 헤더가 정확한 순서와 레벨(# 또는 ##)로 존재하는가? (`# 작업이름`, `## 목표 이름`, `## 진행`, `## 결정`, `## 결과`, `## 다음할일`, `## 산출물`) - '# 작업이름' 부분은 실제 작업 이름으로 대체될 수 있음.
        2.  **Bullet Point 개수:** `## 산출물` 섹션을 제외한 각 `##` 섹션의 bullet point (`- `) 개수가 2개 이상 5개 이하인가? (0개도 허용 안됨. 내용 없을 시 '- 없음.'과 같이 2개 이상으로 맞춰야 함)
        3.  **산출물 형식:** `## 산출물` 섹션에는 (a) 파일 이름들이 쉼표로만 구분되어 나열되어 있거나, (b) **오직 '없음'이라는 두 글자만 정확히 있어야 합니다.** (a)의 경우 bullet point, 경로, 설명 등은 금지됩니다. (b)의 경우 '없음' 외 다른 어떤 텍스트도 없어야 합니다.
        4.  **금지 요소:** 표, 코드 블록(```), 인라인 코드(`), 이미지, HTML 태그, ### 헤더 등 금지된 요소가 포함되지 않았는가? # <-- ### 헤더 금지 추가
        5.  **언어:** 모든 내용이 한국어로 작성되었는가? (코드/파일명, 고유명사 제외)

        ### 검증 대상 Markdown
        ```markdown
        {md}
        ```

        ### 결과 보고
        - 모든 규칙을 완벽하게 준수하면 **오직 'OK'** 라고만 응답합니다. (따옴표 제외)
        - 규칙 위반 사항이 하나라도 발견되면, 'OK' 대신 **발견된 모든 문제점**을 간결하게 한 줄씩 나열합니다. (`-` 사용 가능)
        """)
        res = self._req(sys_msg, user_msg)

        if not res:
            return False, f"AI 검증 응답 없음 ({self.get_name()})"

        # 최종 단순화된 검증 로직 (re 사용) - 변경 없음
        up = res.upper()
        ok  = bool(re.search(r"\bOK\b", up))
        # 부정 토큰 리스트에 '###' 자체를 검사하는 것은 위험할 수 있으므로,
        # 금지 요소 규칙에 '### 헤더'를 명시하고 AI가 판단하도록 유도.
        # bad_token은 기존대로 유지하거나 필요시 더 정교화.
        bad = bool(re.search(r"(❌|NG\b|FAIL|ERROR|문제점|PROBLEM)", up))
        is_ok = ok and not bad

        return is_ok, res

    def load_report(self, md: str) -> str:
        # 변경 없음
        sys_msg = textwrap.dedent("""
        당신은 동료로부터 프로젝트 상태 보고서를 전달받아 내용을 파악해야 하는 개발자입니다.
        주어진 Markdown 보고서를 주의 깊게 읽고, 이해한 내용을 바탕으로 현재 상황과 즉시 해야 할 일을 요약해주세요.
        """)
        user_msg = textwrap.dedent(f"""
        다음은 전달받은 프로젝트 상태 보고서입니다.

        ### CONTEXT START: State from Commit ###
        {md}
        ### CONTEXT END ###

        이 보고서의 내용을 완전히 이해했다면, 아래 두 가지 항목으로 나누어 답변해주세요.

        1.  **현재 상황 요약 (1~2 문장):** 이 프로젝트/작업이 현재 어떤 상태에 있는지 핵심만 간략하게 요약해주세요.
        2.  **즉시 시작할 일 (1~3가지):** 보고서의 '다음할일' 섹션을 바탕으로, 내가 지금 당장 시작해야 할 가장 중요한 작업 1~3가지를 구체적인 행동으로 명시해주세요.

        답변은 다른 설명 없이 위의 두 항목만 명확하게 구분하여 작성합니다.
        """)
        return self._req(sys_msg, user_msg)