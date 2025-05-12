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
            return response.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Ollama API 연결 실패 ({self.base_url}): {e}") from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"Ollama API 응답 시간 초과: {e}") from e
        except requests.exceptions.RequestException as e:
            err_msg = f"Ollama API 요청 실패: {e}"
            if e.response is not None:
                err_msg += f"\n상태 코드: {e.response.status_code}\n응답: {e.response.text[:500]}"
            raise RuntimeError(err_msg) from e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Ollama API 응답 처리 오류 (예상치 못한 형식): {e}") from e

    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        sys_msg = textwrap.dedent("""
        당신은 한국어로 프로젝트 인수인계 문서를 작성하는 매우 정확하고 꼼꼼한 시니어 개발자입니다.
        주어진 모든 규칙을 **단 하나도 빠짐없이, 글자 그대로 정확하게** 준수하여 Markdown 형식으로 문서를 생성해야 합니다.
        특히 헤더의 레벨과 형식이 매우 중요합니다. 이 문서는 다른 동료가 현재 상황을 즉시 파악하고 작업을 이어받는 데 사용됩니다.
        """)
        user_msg = textwrap.dedent(f"""
        ### **매우 중요한 작성 지침 (반드시, 반드시 엄수!)**

        당신은 제공된 정보를 바탕으로 Markdown 형식의 인수인계 문서를 생성해야 합니다.
        아래 **모든 규칙을 하나도 빠짐없이, 글자 그대로 정확하게 지켜서** 작성해주십시오.

        ---
        **1. ✨절대적인 헤더 구조 및 순서 (가장 중요!)✨:**
        아래 제시된 7개의 섹션은 **정확히 이 순서대로, 그리고 명시된 헤더 레벨 및 형식으로만** 작성해야 합니다.
        다른 섹션을 추가하거나, 헤더 레벨을 임의로 변경하거나, 순서를 바꾸는 것은 **절대 금지**입니다.

        * **1-1. 최상위 헤더 (문서 제목):**
            * **반드시, 반드시 `# {task}` 형식이어야 합니다.** (예시: `# 주요 기능 개발`)
            * 이것은 **단 하나의 `#` 심볼**과 한 칸의 공백, 그리고 아래 "입력 정보"에서 제공되는 '작업 이름'으로 구성됩니다.
            * **경고: 절대로 `## {task}` 또는 `### {task}` 와 같이 `#`을 두 개 이상 사용해서는 안 됩니다.** 오직 `#` 하나만 사용해야 합니다.

        * **1-2. 하위 섹션 헤더:**
            * 나머지 6개 섹션(목표, 진행, 결정, 결과, 다음할일, 산출물)은 **반드시 `## 섹션이름` 형식이어야 합니다.** (예시: `## 목표`, `## 진행`)
            * 이것은 **정확히 두 개의 `##` 심볼**과 한 칸의 공백, 그리고 정해진 섹션 이름으로 구성됩니다.

        **올바른 전체 헤더 구조 예시 (이 구조를 반드시 따르세요):**
        ```markdown
        # {task}
        ## 목표
        ## 진행
        ## 결정
        ## 결과
        ## 다음할일
        ## 산출물
        ```
        **(위 예시에서 `{task}` 부분만 실제 작업 이름으로 대체하고, 나머지는 그대로 사용합니다.)**

        ---
        **2. 내용 작성 규칙 - 간결성:**
        * `## 산출물` 섹션을 제외한 각 `##` 섹션에는 핵심 내용을 담은 bullet point (`- ` 형식)를 **정확히 2개에서 5개 사이**로 사용합니다. (1개도 안되고, 6개도 안됩니다.)
        * 문장은 짧고 명확하게, 한국어로 작성하며 불필요한 미사여구나 추측은 배제합니다.

        ---
        **3. 내용 작성 규칙 - `## 산출물` 섹션:**
        * 이 섹션에는 **오직 제공된 '현재 작업 산출물 목록'에 있는 파일 이름들만 쉼표(`,`)로 구분하여 나열**합니다. (예시: `file1.py, image.png, document.pdf`)
        * 산출물 목록이 없다면, 섹션 내용으로 "**없음**" 이라고만 정확히 명시합니다.
        * 파일 경로, 추가 설명, bullet point 등 다른 내용은 **절대 포함하지 마십시오.**

        ---
        **4. 금지 사항 (엄격히 준수):**
        * **어떤 경우에도** 표, 코드 블록(```), 인라인 코드(`), 이미지, HTML 태그 등 기본 Markdown(위에 명시된 헤더, bullet point, 쉼표 구분 목록)을 벗어난 그 어떤 확장 문법도 사용을 **금지**합니다.
        * 이것은 인수인계 문서의 표준 형식을 유지하기 위함입니다.

        ---
        **5. 언어:**
        * 모든 내용은 한국어로 작성합니다. (단, 파일명이나 코드에서 유래한 고유명사는 원본 그대로 사용 가능)

        ---
        ### **입력 정보 (이것을 바탕으로 위 규칙에 맞춰 작성):**

        * **작업 이름 (최상위 헤더 `#` 뒤에 사용될 내용):** `{task}`
            * **다시 한 번 강조합니다: 이 작업 이름은 반드시 `# {task}` 형식으로 문서의 첫 번째 줄, 최상위 헤더로 사용되어야 합니다. `#` 하나입니다!**

        * **최근 작업 내용 / 대화 요약 (Context - 각 `##` 섹션 내용 구성에 참고):**
            ```text
            {ctx or '제공된 내용 없음'}
            ```

        * **현재 작업 산출물 목록 (`## 산출물` 섹션에 사용될 내용):** `{', '.join(arts) if arts else '없음'}`

        ---
        ### **요청: Markdown 출력**
        위의 **모든 지침과 규칙, 특히 절대적인 헤더 구조 규칙(1-1, 1-2)을 철저히 준수하여** 인수인계 문서를 Markdown 형식으로 생성해주십시오.
        생성된 문서의 첫 번째 줄이 정확히 `# {task}` 형식인지 스스로 다시 한번 확인하고 출력해주십시오.
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
        
        if not res:
            return False, f"AI 검증 응답 없음 ({self.get_name()})"

        stripped_res_upper = res.strip().upper()
        is_ok = stripped_res_upper.startswith("**OK**") or stripped_res_upper.startswith("OK")
        
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