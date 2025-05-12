#!/usr/bin/env python3
# backends/ollama.py - Ollama AI Backend (v1.2 - 코드 끝 정리)

import os
import re
import json
import textwrap
import requests
from typing import List, Tuple, Dict, Any
# 상대 경로 임포트 확인: handover.py와 같은 폴더 내 backends 폴더 기준
from .base import AIBaseBackend

class OllamaBackend(AIBaseBackend):
    """Ollama와 상호작용하는 AI 백엔드 구현체."""

    # ---------- 초기화 --------------------------------------------------------
    def __init__(self, config: Dict[str, Any]):
        """
        Ollama 백엔드 초기화.

        Args:
            config: 설정 딕셔너리. 'model', 'ollama_base_url' 키 사용 가능.
                    환경 변수 AI_MODEL, OLLAMA_BASE_URL 또는 기본값 사용.

        Raises:
            ConnectionError: Ollama 서버에 연결할 수 없는 경우.
        """
        super().__init__(config)
        # 모델 이름 설정 (설정값 > 환경변수 > 기본값 'llama3')
        self.model = config.get("model", os.getenv("AI_MODEL", "llama3"))
        # Ollama 서버 URL 설정 (설정값 > 환경변수 > 기본값 'http://localhost:11434')
        self.base_url = config.get(
            "ollama_base_url",
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        # 서버 연결 테스트
        try:
            # HEAD 요청으로 서버 상태 빠르게 확인 (timeout 5초)
            response = requests.head(self.base_url, timeout=5)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        except requests.exceptions.RequestException as e:
            # 연결 실패 또는 HTTP 오류 시 ConnectionError 발생
            raise ConnectionError(
                f"Ollama 서버({self.base_url})에 연결할 수 없습니다: {e}"
            ) from e

    # ---------- 메타 정보 ----------------------------------------------------
    @staticmethod
    def get_name() -> str:
        """백엔드의 고유 이름을 반환합니다."""
        return "ollama"

    @staticmethod
    def get_config_description() -> str:
        """백엔드 설정에 대한 설명을 반환합니다."""
        # 여러 줄 문자열을 깔끔하게 정의
        return textwrap.dedent(
            """
            Ollama 설정:
            - model: (선택) 사용할 Ollama 모델 이름 (환경변수: AI_MODEL, 기본값: 'llama3')
            - ollama_base_url: (선택) Ollama API 서버의 기본 URL (환경변수: OLLAMA_BASE_URL, 기본값: 'http://localhost:11434')
            """
        )

    # ---------- 내부 API 요청 래퍼 ------------------------------------------
    def _req(self, sys_msg: str, user_msg: str) -> str:
        """
        Ollama chat API에 요청을 보내고 응답 텍스트를 반환하는 내부 헬퍼 메서드.

        Args:
            sys_msg: 시스템 메시지 내용.
            user_msg: 사용자 메시지 내용.

        Returns:
            Ollama 모델의 응답 텍스트.

        Raises:
            RuntimeError: API 호출 중 오류 발생 시.
        """
        api_url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            "options": {
                "temperature": 0, # 결정론적 응답을 위해 온도 0 설정
                "num_ctx": 4096,  # 컨텍스트 길이 설정 (필요시 조정)
            },
            "stream": False, # 스트리밍 응답 비활성화
        }
        try:
            # POST 요청 실행 (timeout 180초)
            response = requests.post(api_url, json=payload, timeout=180)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생

            # 응답 JSON 파싱 및 메시지 내용 추출
            response_data = response.json()
            message_content = (response_data.get("message") or {}).get("content", "")

            # 응답이 비어있는 경우 처리 (모델이 빈 응답을 줄 수도 있음)
            if not message_content:
                 print(f"[WARN] Ollama model '{self.model}' returned an empty response.")
                 return "" # 빈 문자열 반환

            return message_content.strip() # 앞뒤 공백 제거 후 반환

        except requests.exceptions.Timeout:
             raise RuntimeError(f"Ollama API 호출 시간 초과 ({api_url})")
        except requests.exceptions.RequestException as e:
            # 네트워크 오류, HTTP 오류 등 포함
            raise RuntimeError(f"Ollama API 호출 실패 ({api_url}): {e}") from e
        except json.JSONDecodeError:
             raise RuntimeError(f"Ollama API 응답 JSON 파싱 실패 ({api_url}). 응답 내용: {response.text[:500]}...") # 응답 일부 표시
        except Exception as e:
            # 기타 예기치 않은 오류
            raise RuntimeError(f"Ollama API 요청 중 예기치 않은 오류 발생: {e}") from e

    # ---------- 1) 인수인계 문서 생성 ----------------------------------------
    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        """
        주어진 정보를 바탕으로 표준 형식의 Markdown 인수인계 문서를 생성합니다.

        Args:
            task: 작업 이름 (문서의 메인 제목으로 사용됨).
            ctx: 최근 작업 내용 요약 (진행, 결정, 결과 등의 내용을 유추하는 데 사용됨).
            arts: 현재 작업 산출물 파일 이름 목록.

        Returns:
            생성된 Markdown 인수인계 문서 문자열.
        """
        # AI에게 전달할 상세 지침 (절대 규칙)
        # f-string 내에서 중괄호 자체를 사용하려면 {{ }} 사용
        guidelines = textwrap.dedent(
            f"""
            ### 절대 규칙 (반드시 지킬 것)
            1.  **헤더 구조 및 순서 고정 (정확히 7개):**
                ```markdown
                # {task}
                ## 목표
                ## 진행
                ## 결정
                ## 결과
                ## 다음할일
                ## 산출물
                ```
            2.  **각 섹션 내용:**
                * `## 목표`, `## 진행`, `## 결정`, `## 결과`, `## 다음할일` 섹션은 각각 2개 이상 5개 이하의 한국어 불릿 포인트(`- `)로 작성해야 하며, 각 문장은 마침표(`.`)로 끝나야 합니다. 내용은 제공된 '최근 작업 요약'을 기반으로 추론하여 작성합니다.
                * `## 산출물` 섹션은 제공된 '현재 작업 산출물' 목록을 쉼표(`, `)로 구분하여 나열합니다. 산출물이 없으면 정확히 '없음' 이라는 단어 하나만 적습니다.
            3.  **형식 제한:**
                * Markdown 테이블, 코드 블록(```), 인라인 코드(`), HTML 태그, 이미지, 체크박스([-] [x]) 사용을 절대 금지합니다.
                * 오직 지정된 헤더와 불릿 포인트, 일반 텍스트만 사용합니다.
            4.  **언어:**
                * 모든 내용은 한국어로 작성합니다. (단, Oracle, PostgreSQL, Python 등과 같은 고유명사나 기술 용어는 예외적으로 영어 사용 가능)
            5.  **출력:**
                * 최종 응답은 위 규칙에 따라 생성된 Markdown 문서 내용만 포함해야 합니다. 이 지침 자체나 다른 부가 설명은 절대 포함하지 마십시오.
            """
        )

        # 시스템 메시지: AI의 역할과 지침 명시
        sys_msg = (
            "당신은 경험 많은 시니어 개발자입니다. 주어진 정보를 바탕으로, 아래 제공된 '절대 규칙'을 **반드시** 준수하여 "
            "프로젝트 인수인계 문서를 Markdown 형식으로 작성해야 합니다.\n\n"
            + guidelines
        )

        # 사용자 메시지: AI에게 필요한 정보 제공
        arts_str = ", ".join(arts) if arts else "없음" # 산출물 목록 문자열 처리
        user_msg = textwrap.dedent(
            f"""
            다음 정보를 사용하여 인수인계 문서를 생성하십시오.

            **작업 이름:** {task}

            **최근 작업 요약:**
            ```
            {ctx or '제공된 작업 요약 내용 없음.'}
            ```

            **현재 작업 산출물:** {arts_str}

            **요청:** 위 정보를 바탕으로, 앞서 제시된 '절대 규칙'에 맞는 Markdown 인수인계 문서를 생성하여 그 내용만 응답으로 반환하십시오.
            """
        )

        # Ollama API 호출하여 문서 생성 요청
        return self._req(sys_msg, user_msg)

    # ---------- 2) 생성 문서 검증 (개선됨) -----------------------------------
    def verify_summary(self, md: str) -> Tuple[bool, str]:
        """
        AI가 생성한 Markdown 인수인계 문서가 지정된 규칙을 준수하는지 검증합니다.

        Args:
            md: 검증할 Markdown 문서 내용.

        Returns:
            Tuple[bool, str]: (검증 통과 여부, 결과 메시지).
                              성공 시 메시지는 "AI validation passed."
                              실패 시 메시지는 실패 사유.
        """
        # 시스템 메시지: 검증기의 역할과 규칙, 응답 형식 명시
        sys_msg = textwrap.dedent("""
            당신은 특정 형식의 Markdown 인수인계 문서를 위한 매우 엄격한 검증기입니다.

            **검증 규칙:**
            1.  정확히 7개의 헤더가 다음 순서대로 존재해야 합니다: `# 제목`, `## 목표`, `## 진행`, `## 결정`, `## 결과`, `## 다음할일`, `## 산출물`
            2.  첫 번째 헤더는 정확히 `# `로 시작해야 합니다.
            3.  헤더 2부터 7까지는 정확히 `## 헤더이름 ` 형식으로 시작해야 합니다 (헤더 이름 뒤 공백 포함).
            4.  `## 목표`, `## 진행`, `## 결정`, `## 결과`, `## 다음할일` 섹션은 각각 2개 이상 5개 이하의 불릿 포인트(`- `)를 포함해야 하며, 각 불릿 포인트는 마침표(`.`)로 끝나는 완전한 한국어 문장이어야 합니다.
            5.  `## 산출물` 섹션은 쉼표(`, `)로 구분된 파일 이름 목록이거나, 정확히 '없음'이라는 단어 하나만 포함해야 합니다.
            6.  Markdown 테이블, 코드 블록(```), 인라인 코드(`), HTML 태그, 이미지, 체크박스([-] [x])는 절대 허용되지 않습니다.
            7.  모든 내용은 한국어로 작성되어야 합니다 (단, Oracle, Python 같은 고유명사 제외).

            **작업:**
            제공된 Markdown 문서를 위 규칙에 따라 분석하십시오.

            **응답 형식 (다음 두 가지 중 하나로만 응답):**
            * 모든 규칙을 완벽하게 만족하는 경우: `Validation: PASS`
            * 하나 이상의 규칙이라도 위반하는 경우: `Validation: FAIL - [구체적인 실패 사유 목록]` (실패 사유는 간결하게 요점만 나열)

            **주의:** 응답에는 검증 대상 Markdown 원본을 포함하지 마십시오. 오직 위에서 명시한 "Validation: PASS" 또는 "Validation: FAIL - ..." 형식 중 하나로만 응답해야 합니다.
            """)

        # 사용자 메시지: 검증 대상 Markdown 전달
        user_msg = f"다음 Markdown 문서를 검증하십시오:\n\n```markdown\n{md}\n```"

        # Ollama API 호출하여 검증 요청
        res = self._req(sys_msg, user_msg)

        # 응답이 비어있는 경우 처리
        if not res:
            return False, "AI 검증 응답이 비어있습니다."

        # 대소문자 구분 없이 비교하기 위해 대문자로 변환
        res_upper = res.upper()

        # 응답 형식 확인 및 결과 처리
        if res_upper.startswith("VALIDATION: PASS"):
            # "PASS"로 시작하는 경우, 만일을 위해 실패 관련 키워드가 없는지 추가 확인
            if not re.search(r"(FAIL|ERROR|PROBLEM|ISSUE|VIOLAT|INVALID|INCORRECT|MISSING)", res_upper):
                 # 실패 키워드가 없으면 최종 성공으로 간주
                 return True, "AI validation passed." # 내부 성공 메시지 반환
            else:
                 # "PASS"로 시작했지만 실패 키워드가 포함된 경우, 실패로 처리
                 failure_reason = res[len("Validation: PASS"):].strip() # PASS 이후 내용 추출
                 return False, f"AI 검증이 PASS로 시작했지만 실패 표시자를 포함: {failure_reason or res}"

        elif res_upper.startswith("VALIDATION: FAIL"):
            # "FAIL"로 시작하는 경우, 실패 사유 추출
            reason = res[len("Validation: FAIL"):].strip()
            # 실패 사유가 '-'로 시작하면 제거 (예: "FAIL - - 헤더...")
            if reason.startswith("-"):
                reason = reason[1:].strip()
            # 추출된 사유가 있으면 반환, 없으면 기본 실패 메시지 반환
            return False, reason if reason else "AI 검증 실패 (구체적인 사유 없음)"
        else:
            # 응답이 예상된 "Validation: PASS" 또는 "Validation: FAIL"로 시작하지 않는 경우
            return False, f"AI 검증 응답이 예상된 형식(PASS/FAIL)이 아님: {res}"


    # ---------- 3) 로드된 보고서 요약 -----------------------------------------
    def load_report(self, md: str) -> str:
        """
        과거 저장된 인수인계 문서(Markdown)를 AI가 이해하고 요약 보고서를 생성합니다.

        Args:
            md: 로드된 Markdown 인수인계 문서 내용.

        Returns:
            AI가 생성한 요약 보고서 문자열 (현재 상황 요약, 다음 할 일 포함).
        """
        # 시스템 메시지: AI의 역할 정의
        sys_msg = (
            "당신은 프로젝트 관리자입니다. 전달받은 Markdown 형식의 프로젝트 인수인계 보고서를 분석하고, "
            "현재 상황과 다음 단계를 명확하게 요약해야 합니다."
        )

        # 사용자 메시지: 보고서 내용과 요약 요청 구체화
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

        # Ollama API 호출하여 요약 보고서 생성 요청
        return self._req(sys_msg, user_msg)

# 파일 끝에 빈 줄 추가 (일부 린터 호환성)

```

이 업데이트된 코드를 `backends/ollama.py`에 적용하고, 위에서 안내한 해결 방법 중 하나를 선택하여 다시 시도해 보시기 바랍