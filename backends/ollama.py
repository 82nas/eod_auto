# ollama.py
import os
import requests
import textwrap
import json # For potential JSONDecodeError
from typing import List, Tuple, Dict, Any
from .base import AIBaseBackend # Use relative import

class OllamaBackend(AIBaseBackend):
    """AI Backend implementation for Ollama."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", os.getenv("AI_MODEL", "llama3"))
        self.base_url = config.get("ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        # Check Ollama availability at init
        try:
            # Use HEAD request for quick check without transferring body
            response = requests.head(self.base_url, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # Provide more specific error guidance
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
                    "options": {"temperature": 0, "num_ctx": 4096}, # Consistent results
                    "stream": False
                },
                timeout=180
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Ollama API 연결 실패 ({self.base_url}): {e}") from e
        except requests.exceptions.Timeout as e:
             raise RuntimeError(f"Ollama API 응답 시간 초과: {e}") from e
        except requests.exceptions.RequestException as e: # Catch other request errors (like HTTPError)
            err_msg = f"Ollama API 요청 실패: {e}"
            if e.response is not None:
                err_msg += f"\n상태 코드: {e.response.status_code}\n응답: {e.response.text[:500]}" # Show partial response
            raise RuntimeError(err_msg) from e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            # Handle cases where response structure is not as expected
            raise RuntimeError(f"Ollama API 응답 처리 오류 (예상치 못한 형식): {e}") from e

    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        sys_msg = textwrap.dedent("""
        당신은 한국어로 프로젝트 인수인계 문서를 작성하는 시니어 개발자입니다.
        다음 규칙을 **반드시** 준수하여 간결하고 명확하게 Markdown 형식으로 작성해주세요.
        목표는 이 문서를 통해 다른 동료가 현재 상황을 빠르게 파악하고 다음 작업을 이어받을 수 있도록 하는 것입니다.
        """)
        user_msg = textwrap.dedent(f"""
        ### 작성 규칙
        1.  **정확한 순서와 헤더:** 아래 7개의 섹션을 **정확히 이 순서대로**, **주어진 헤더 형식(# 또는 ##)**을 사용하여 작성해야 합니다. 다른 섹션 추가 금지.
            ```markdown
            # {task}
            ## 목표
            ## 진행
            ## 결정
            ## 결과
            ## 다음할일
            ## 산출물
            ```
        2.  **간결성:** 각 섹션(`산출물` 제외)에는 핵심 내용만 담은 bullet point (`- `)를 2~5개 사용합니다. 문장은 짧고 명확하게 작성하고, 불필요한 미사여구나 추측은 배제합니다.
        3.  **산출물 섹션:** 이 섹션에는 제공된 산출물 목록(`{', '.join(arts) or '없음'}`)에 있는 파일 이름만 `, `로 구분하여 나열합니다. 파일 경로, 설명, 추가 bullet point는 사용하지 않습니다. 목록이 없으면 '없음'이라고 명시합니다.
        4.  **금지 사항:** 표, 코드 블록(```), 인라인 코드(`), 이미지, HTML 태그 등 Markdown 확장 문법 사용을 금지합니다. 오직 기본 Markdown 문법(헤더, bullet point)만 사용하세요.
        5.  **언어:** 모든 내용은 한국어로 작성합니다.

        ### 입력 정보
        - **작업 이름:** {task} (이것을 최상위 헤더 `#` 뒤에 사용)
        - **최근 작업 내용 / 대화 요약 (Context):**
          ```
          {ctx or '제공된 내용 없음'}
          ```
        - **현재 작업 산출물 목록:** {', '.join(arts) or '없음'} (이것을 `## 산출물` 섹션에 나열)

        ### 출력 (Markdown)
        위 규칙과 입력 정보를 바탕으로 인수인계 문서를 작성하세요.
        """)
        return self._req(sys_msg, user_msg)

    def verify_summary(self, md: str) -> Tuple[bool, str]:
        sys_msg = textwrap.dedent("""
        당신은 Markdown 문서 형식을 검증하는 매우 꼼꼼한 검토자입니다.
        주어진 Markdown 텍스트가 아래 규칙을 모두 준수하는지 확인하고 결과를 보고합니다.
        """)
        user_msg = textwrap.dedent(f"""
        ### 검증 규칙
        1.  **필수 헤더 및 순서:** 다음 7개 헤더가 정확한 순서와 레벨(# 또는 ##)로 존재하는가? (`# 작업이름`, `## 목표`, `## 진행`, `## 결정`, `## 결과`, `## 다음할일`, `## 산출물`) - '# 작업이름' 부분은 실제 작업 이름으로 대체될 수 있음.
        2.  **Bullet Point 개수:** `## 산출물` 섹션을 제외한 각 `##` 섹션의 bullet point (`- `) 개수가 2개 이상 5개 이하인가? (0개도 허용 안됨)
        3.  **산출물 형식:** `## 산출물` 섹션에는 파일 이름만 쉼표로 구분되어 나열되어 있는가? 또는 '없음' 텍스트만 있는가? (bullet point, 경로, 설명 금지)
        4.  **금지 요소:** 표, 코드 블록(```), 인라인 코드(`), 이미지, HTML 태그 등 금지된 요소가 포함되지 않았는가?
        5.  **언어:** 모든 내용이 한국어로 작성되었는가? (코드/파일명 제외)

        ### 검증 대상 Markdown
        ```markdown
        {md}
        ```

        ### 결과 보고
        - 모든 규칙을 완벽하게 준수하면 **오직 'OK'** 라고만 응답합니다.
        - 규칙 위반 사항이 **하나라도** 발견되면, 'OK' 대신 **발견된 모든 문제점**을 간결하게 한 줄씩 나열하여 보고합니다. (예: "- ## 결정 섹션 bullet point 1개 (최소 2개 필요)\n- ## 산출물 섹션에 설명 포함됨")
        """)
        res = self._req(sys_msg, user_msg)
        # Basic check for empty response which might mean failure
        if not res:
            return False, "AI 검증 응답 없음"
        is_ok = res.strip().upper() == "OK"
        return is_ok, res

    def load_report(self, md: str) -> str:
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