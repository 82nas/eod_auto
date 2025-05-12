# backends/ollama.py
import os, re, json, textwrap, requests
from typing import List, Tuple, Dict, Any
from .base import AIBaseBackend


class OllamaBackend(AIBaseBackend):
    """AI Backend implementation for Ollama."""

    # ---------- 초기화 --------------------------------------------------------
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

    # ---------- 메타 ---------------------------------------------------------
    @staticmethod
    def get_name() -> str:
        return "ollama"

    @staticmethod
    def get_config_description() -> str:
        return textwrap.dedent(
            """
            Ollama 설정:
            - model: (선택) Ollama 모델 이름 (기본값: AI_MODEL 또는 'llama3')
            - ollama_base_url: (선택) Ollama 서버 URL
            """
        )

    # ---------- 내부 요청 래퍼 -----------------------------------------------
    def _req(self, sys_msg: str, user_msg: str) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "options": {"temperature": 0, "num_ctx": 4096},
                    "stream": False,
                },
                timeout=180,
            )
            resp.raise_for_status()
            return (
                (resp.json().get("message") or {}).get("content", "").strip()
            )
        except Exception as e:
            raise RuntimeError(f"Ollama 호출 실패: {e}") from e

    # ---------- 1) 인수인계 문서 생성 ----------------------------------------
    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        guidelines = textwrap.dedent(
            f"""
            ### 절대 규칙
            • 헤더 구조(정확히 7개, 순서 고정)
              # {task}
              ## 목표 이름
              ## 진행
              ## 결정
              ## 결과
              ## 다음할일
              ## 산출물
            • 각 ##(산출물 제외) → bullet 2~5개, 마침표로 끝나는 한국어 문장
            • ## 산출물 → 파일명 쉼표 나열, 없으면 '없음' 단어 하나
            • 금지: 표·코드블록·인라인코드·HTML·이미지·체크박스
            • 영어 사용 금지(Oracle·PostgreSQL 등 고유명사 제외)
            • 위 지침을 **본문에 절대 포함하지 말 것**
            """
        )

        sys_msg = (
            "당신은 시니어 개발자입니다. 아래 지침을 철저히 지키며 Markdown "
            "인수인계 문서를 생성하세요.\n"
            + guidelines
        )

        arts_str = ", ".join(arts) if arts else "없음"
        user_msg = textwrap.dedent(
            f"""
            작업 이름: {task}

            최근 작업 요약:
            {ctx or '제공된 내용 없음'}

            현재 작업 산출물: {arts_str}

            위 정보를 반영해 7-헤더 인수인계 문서만 반환하십시오.
            """
        )

        return self._req(sys_msg, user_msg)

    # ---------- 2) 생성 문서 검증 -------------------------------------------
    def verify_summary(self, md: str) -> Tuple[bool, str]:
        sys_msg = (
            "당신은 Markdown 형식 검사기입니다. 규칙 불일치 사항을 찾으면 나열하고, "
            "완벽히 통과하면 OK만 반환하세요."
        )
        # 규칙 목록(헤더 등)은 system 메시지에 이미 포함돼 있으므로 user_msg 로 문서 전달
        user_msg = f"검증 대상 Markdown:\n```markdown\n{md}\n```"

        res = self._req(sys_msg, user_msg)
        if not res:
            return False, "AI 검증 응답 없음"

        up = res.upper()
        ok_token = bool(re.search(r"\bOK\b", up))
        bad_token = bool(re.search(r"(❌|NG\b|FAIL|ERROR|문제점|PROBLEM)", up))
        return ok_token and not bad_token, res

    # ---------- 3) 보고서 요약 ----------------------------------------------
    def load_report(self, md: str) -> str:
        sys_msg = "전달받은 프로젝트 보고서를 요약해 주세요."
        user_msg = (
            f"### 보고서\n{md}\n\n"
            "1) 현재 상황 요약(1-2문장)\n2) 즉시 해야 할 일 1-3가지"
        )
        return self._req(sys_msg, user_msg)
