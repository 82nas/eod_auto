:: setup_handover.cmd  (아이콘/이모지 제거 버전)
@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo === handover one-key setup ===

python -m venv .venv
call ".\.venv\Scripts\activate.bat"

python -m pip install --upgrade pip
pip install gitpython requests rich python-dotenv markdown2 ruff black pre-commit

mkdir artifacts  2>NUL
mkdir backends    2>NUL
mkdir ai_states   2>NUL
if not exist backends\__init__.py type NUL > backends\__init__.py

ollama pull llama3

set "repoOwner=82nas"
set "repoName=eod_auto"
set "branch=main"
set "raw=https://raw.githubusercontent.com/%repoOwner%/%repoName%/%branch%"

curl -L "%raw%/handover.py"                 -o handover.py
curl -L "%raw%/backends/base.py"            -o backends\base.py
curl -L "%raw%/backends/ollama.py"          -o backends\ollama.py
curl -L "%raw%/backends/huggingface.py"     -o backends\huggingface.py

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

echo === 초기 테스트 ===
python handover.py --help

echo.
echo *** 설치 완료! 가상환경 활성화 명령 ***
echo     .\.venv\Scripts\activate.bat
pause
ENDLOCAL
