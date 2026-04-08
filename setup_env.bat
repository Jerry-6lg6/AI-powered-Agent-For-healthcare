@echo off
REM =========================================================
REM setup_env.bat - AI-Powered Healthcare Agent
REM Usage:
REM   setup_env.bat         -> CUDA 12.1 (default)
REM   setup_env.bat cu118   -> CUDA 11.8
REM   setup_env.bat cpu     -> CPU only
REM =========================================================

SET CUDA_TAG=%1
IF "%CUDA_TAG%"=="" SET CUDA_TAG=cu121
SET VENV_DIR=.venv

REM -- Step 0: Check Python version (requires 3.9 to 3.11) --
FOR /F "tokens=2 delims= " %%V IN ('python --version 2^>^&1') DO SET PY_VERSION=%%V
FOR /F "tokens=1,2 delims=." %%A IN ("%PY_VERSION%") DO (
  SET PY_MAJOR=%%A
  SET PY_MINOR=%%B
)

IF %PY_MAJOR% LSS 3 GOTO :bad_python
IF %PY_MINOR% LSS 9 GOTO :bad_python
IF %PY_MINOR% GTR 12 GOTO :bad_python
ECHO [OK] Python %PY_VERSION%
GOTO :continue

:bad_python
ECHO [ERROR] Python %PY_VERSION% detected.
ECHO         This project requires Python 3.9 to 3.11.
ECHO         Please install a compatible version from https://www.python.org/downloads/
EXIT /B 1

:continue
REM -- Step 1: Create virtual environment --
IF NOT EXIST "%VENV_DIR%" (
  ECHO Creating virtual environment in %VENV_DIR%
  python -m venv %VENV_DIR%
  IF ERRORLEVEL 1 (
    ECHO [ERROR] Failed to create virtual environment.
    EXIT /B 1
  )
) ELSE (
  ECHO [OK] Virtual environment already exists, skipping creation.
)

CALL %VENV_DIR%\Scripts\activate.bat
IF ERRORLEVEL 1 (
  ECHO [ERROR] Failed to activate virtual environment.
  EXIT /B 1
)

python -m pip install --upgrade pip --quiet

REM -- Step 2: Install PyAudio directly (no pipwin needed for Python 3.9+) --
pip show pyaudio >nul 2>&1
IF ERRORLEVEL 1 (
  ECHO Installing PyAudio
  pip install PyAudio --quiet
  IF ERRORLEVEL 1 (
    ECHO [ERROR] PyAudio installation failed.
    ECHO         Try manually: pip install PyAudio
    ECHO         Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
    EXIT /B 1
  )
  ECHO [OK] PyAudio installed.
) ELSE (
  ECHO [OK] PyAudio already installed, skipping.
)

REM -- Step 3: Install PyTorch --
IF "%CUDA_TAG%"=="cpu" (
  ECHO Installing PyTorch CPU-only
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
) ELSE (
  ECHO Installing PyTorch with %CUDA_TAG%
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA_TAG% --quiet
)
IF ERRORLEVEL 1 (
  ECHO [ERROR] Failed to install PyTorch.
  EXIT /B 1
)
ECHO [OK] PyTorch installed.

REM -- Step 4: Install project dependencies --
IF NOT EXIST "requirements.txt" (
  ECHO [ERROR] requirements.txt not found.
  ECHO         Run this script from the project root folder.
  EXIT /B 1
)
ECHO Installing project dependencies from requirements.txt
pip install -r requirements.txt --quiet
IF ERRORLEVEL 1 (
  ECHO [ERROR] Failed to install some dependencies.
  EXIT /B 1
)
ECHO [OK] All dependencies installed.

REM -- Done --
ECHO.
ECHO [OK] Setup complete!
ECHO      Activate the environment with:
ECHO        %VENV_DIR%\Scripts\activate
ECHO      Then run:
ECHO        python task.py
