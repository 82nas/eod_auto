# backends/huggingface.py
import os
import requests
import textwrap
import json
from typing import List, Tuple, Dict, Any
from .base import AIBaseBackend  # Relative import


class HuggingFaceBackend(AIBaseBackend):
    """AI Backend implementation for Hugging Face Inference API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Suggest a more common/capable default if AI_MODEL not set
        default_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.model = config.get("model", os.getenv("AI_MODEL", default_model))
        self.hf_key = config.get("hf_api_key", os.getenv("HF_API_KEY"))
        if not self.hf_key:
            raise ValueError(
                "Hugging Face API 키 (HF_API_KEY 환경변수 또는 설정)가 필요합니다."
            )
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        # Add a check if model exists or wait for first request? Maybe wait.

    @staticmethod
    def get_name() -> str:
        return "huggingface"

    @staticmethod
    def get_config_description() -> str:
        return textwrap.dedent(
            """
            Hugging Face 설정:
            - model: (선택) Hugging Face 모델 ID (기본값: AI_MODEL 환경변수 또는 'meta-llama/Meta-Llama-3-8B-Instruct').
            - hf_api_key: (필수) Hugging Face API 키 (기본값: HF_API_KEY 환경변수).
        """
        )

    def _req(self, sys_msg: str, user_msg: str) -> str:
        """Sends request to Hugging Face Inference API."""
        headers = {"Authorization": f"Bearer {self.hf_key}"}
        # Use a format suitable for instruction-tuned models like Llama3
        # Reference: https://huggingface.co/blog/llama3#how-to-prompt-llama-3
        # Simplified prompt structure for basic request
        # Note: HF Inference API might have different optimal formats
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1536,  # Allow longer output
                "temperature": 0.1,  # Low temperature for consistency
                "return_full_text": False,  # Get only the generated part
                # "stop_sequences": ["<|eot_id|>"] # Optional: Stop generation explicitly
            },
        }
        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=300
            )
            response.raise_for_status()
            result = response.json()

            # Handle potential variations in HF API response format
            if isinstance(result, list) and result:
                generated_text = result[0].get("generated_text")
            elif isinstance(result, dict):
                # Sometimes the text might be nested differently
                generated_text = result.get("generated_text")
            else:
                generated_text = None

            if not generated_text:
                raise ValueError(
                    f"HF API 응답에서 'generated_text'를 찾을 수 없습니다. 응답: {result}"
                )

            # Clean up response if needed (remove stop tokens etc.)
            if isinstance(generated_text, str):
                # Simple strip for common stop tokens if not handled by API
                generated_text = generated_text.replace("<|eot_id|>", "").strip()

            return generated_text

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"HF API 연결 실패: {e}") from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"HF API 응답 시간 초과: {e}") from e
        except requests.exceptions.RequestException as e:
            err_msg = f"HF API 요청 실패: {e}"
            if e.response is not None:
                # Include potentially useful error info from HF response
                try:
                    err_details = e.response.json()
                    err_msg += f"\n상태 코드: {e.response.status_code}\n오류: {err_details.get('error', e.response.text[:500])}"
                except json.JSONDecodeError:
                    err_msg += f"\n상태 코드: {e.response.status_code}\n응답 (텍스트): {e.response.text[:500]}"
            raise RuntimeError(err_msg) from e
        except (
            ValueError,
            json.JSONDecodeError,
        ) as e:  # Catch JSON parsing or value errors
            raise RuntimeError(f"HF API 응답 처리 오류: {e}") from e

    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        # Identical prompt construction logic as OllamaBackend.make_summary
        sys_msg = textwrap.dedent(
            """
        당신은 한국어로 프로젝트 인수인계 문서를 작성하는 시니어 개발자입니다.
        다음 규칙을 **반드시** 준수하여 간결하고 명확하게 Markdown 형식으로 작성해주세요.
        목표는 이 문서를 통해 다른 동료가 현재 상황을 빠르게 파악하고 다음 작업을 이어받을 수 있도록 하는 것입니다.
        """
        )
        user_msg = textwrap.dedent(
            f"""
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
        """
        )
        return self._req(sys_msg, user_msg)

    def verify_summary(self, md: str) -> Tuple[bool, str]:
        # Identical prompt construction logic as OllamaBackend.verify_summary
        sys_msg = textwrap.dedent(
            """
        당신은 Markdown 문서 형식을 검증하는 매우 꼼꼼한 검토자입니다.
        주어진 Markdown 텍스트가 아래 규칙을 모두 준수하는지 확인하고 결과를 보고합니다.
        """
        )
        user_msg = textwrap.dedent(
            f"""
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
        """
        )
        res = self._req(sys_msg, user_msg)

        if not res:  # AI 응답이 아예 없는 경우
            return False, f"AI 검증 응답 없음 ({self.get_name()})"  # 백엔드 이름 포함

        # *** 수정된 로직 시작 ***
        stripped_res_upper = res.strip().upper()
        # 응답의 시작 부분이 "**OK**" 또는 "OK"이면 성공으로 간주
        is_ok = stripped_res_upper.startswith(
            "**OK**"
        ) or stripped_res_upper.startswith("OK")
        # *** 수정된 로직 끝 ***

        return is_ok, res  # 원래 메시지(res)는 그대로 반환

    def load_report(self, md: str) -> str:
        # Identical prompt construction logic as OllamaBackend.load_report
        sys_msg = textwrap.dedent(
            """
        당신은 동료로부터 프로젝트 상태 보고서를 전달받아 내용을 파악해야 하는 개발자입니다.
        주어진 Markdown 보고서를 주의 깊게 읽고, 이해한 내용을 바탕으로 현재 상황과 즉시 해야 할 일을 요약해주세요.
        """
        )
        user_msg = textwrap.dedent(
            f"""
        다음은 전달받은 프로젝트 상태 보고서입니다.

        ### CONTEXT START: State from Commit ###
        {md}
        ### CONTEXT END ###

        이 보고서의 내용을 완전히 이해했다면, 아래 두 가지 항목으로 나누어 답변해주세요.

        1.  **현재 상황 요약 (1~2 문장):** 이 프로젝트/작업이 현재 어떤 상태에 있는지 핵심만 간략하게 요약해주세요.
        2.  **즉시 시작할 일 (1~3가지):** 보고서의 '다음할일' 섹션을 바탕으로, 내가 지금 당장 시작해야 할 가장 중요한 작업 1~3가지를 구체적인 행동으로 명시해주세요.

        답변은 다른 설명 없이 위의 두 항목만 명확하게 구분하여 작성합니다.
        """
        )
        return self._req(sys_msg, user_msg)
