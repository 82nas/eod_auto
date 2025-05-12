:: ─────────────────────────────────────────────────────────────
:: 📦 handover v1.1 (Integrated) one-key setup – CMD 버전
:: 복사 → 새 파일 setup_handover.cmd 로 저장 → 더블클릭
:: (관리자 권한 불필요, Windows 10/11 기본 명령만 사용)
:: ─────────────────────────────────────────────────────────────
@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo 📦 handover v1.1 (Integrated) one-key setup 시작

:: 1. Python 가상 환경 생성 및 활성화
python -m venv .venv
call ".\.venv\Scripts\activate.bat"

:: 2. pip 업그레이드 + 필수 라이브러리 설치
pip install --upgrade pip
pip install gitpython requests rich python-dotenv markdown2 ruff black pre-commit

:: 3. 폴더/파일 기본 구조
mkdir artifacts 2>NUL
mkdir backends   2>NUL
mkdir ai_states  2>NUL
if not exist backends\__init__.py type NUL > backends\__init__.py

:: 4. Ollama 모델 다운로드 (Ollama CLI 설치·실행 중이어야 함)
ollama pull llama3

:: 5. GitHub 원본 스크립트 다운로드
set "repoOwner=82nas"
set "repoName=eod_auto"
set "branch=main"
set "baseUrl=https://raw.githubusercontent.com/%repoOwner%/%repoName%/%branch%"

curl -L "%baseUrl%/handover.py"                 -o handover.py
curl -L "%baseUrl%/backends/base.py"            -o backends\base.py
curl -L "%baseUrl%/backends/ollama.py"          -o backends\ollama.py
curl -L "%baseUrl%/backends/huggingface.py"     -o backends\huggingface.py

:: 6. pre-commit 설정 파일 작성
(
echo repos:
echo ^- repo: https://github.com/astral-sh/ruff-pre-commit
echo ^  rev: v0.4.5
echo ^  hooks:
echo ^    ^- id: ruff
echo ^- repo: https://github.com/psf/black
echo ^  rev: 24.4.2
echo ^  hooks:
echo ^    ^- id: black
) > .pre-commit-config.yaml
pre-commit install

:: 7. 기본 실행 테스트
echo ⚙️  handover.py --help 테스트…
python handover.py --help

:: 8. 완료 메시지
echo(
echo ✅ 설치 완료!
echo 💡 앞으로 이 폴더에서 작업할 때는 가상 환경부터 활성화하세요:
echo    .\.venv\Scripts\activate.bat
echo(
pause
ENDLOCAL
