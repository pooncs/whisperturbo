@echo off
setlocal enabledelayedexpansion

:: WhisperTurbo Launcher Script
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%"
cd /d "%PROJECT_DIR%" 2>nul

:: Default settings
set "PYTHON_EXE=python"
set "HF_TOKEN="
set "NO_GUI=0"
set "NO_DIARIZATION=0"
set "GUI_PORT=5006"
set "CHECK_MODELS=1"
set "OPEN_BROWSER=1"

:: Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--no-gui" set "NO_GUI=1" & shift & goto parse_args
if /i "%~1"=="--no-diarization" set "NO_DIARIZATION=1" & shift & goto parse_args
if /i "%~1"=="--port" set "GUI_PORT=%~2" & shift & shift & goto parse_args
if /i "%~1"=="--token" set "HF_TOKEN=%~2" & shift & shift & goto parse_args
if /i "%~1"=="--skip-models" set "CHECK_MODELS=0" & shift & goto parse_args
if /i "%~1"=="--no-browser" set "OPEN_BROWSER=0" & shift & goto parse_args
if /i "%~1"=="-h" goto help
if /i "%~1"=="--help" goto help
echo Unknown option: %~1
goto help

:end_parse

:: Print banner
echo ============================================================
echo  WhisperTurbo Launcher
echo  Real-Time Korean -> English Speech Translation
echo ============================================================
echo.

:: Check Python installation
echo [+] Checking Python installation...
%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo [-] Python not found. Please install Python 3.9 or higher.
    echo    Download from: https://www.python.org/downloads/
    exit /b 1
)
for /f "tokens=2" %%v in ('%PYTHON_EXE% --version 2^>^&1') do set "PYTHON_VERSION=%%v"
echo     Python version: %PYTHON_VERSION%

:: Check Python version (3.9+)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set "PYTHON_MAJOR=%%a"
    set "PYTHON_MINOR=%%b"
)
if %PYTHON_MAJOR% LSS 3 goto python_old
if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 9 goto python_old
goto python_ok

:python_old
echo [-] Python 3.9 or higher is required.
exit /b 1

:python_ok

:: Check dependencies
echo [+] Checking dependencies...
set "MISSING_DEPS="
%PYTHON_EXE% -c "import faster_whisper" 2>nul || set "MISSING_DEPS=%MISSING_DEPS% faster-whisper"
%PYTHON_EXE% -c "import sounddevice" 2>nul || set "MISSING_DEPS=%MISSING_DEPS% sounddevice"
%PYTHON_EXE% -c "import torch" 2>nul || set "MISSING_DEPS=%MISSING_DEPS% torch"
%PYTHON_EXE% -c "import panel" 2>nul || set "MISSING_DEPS=%MISSING_DEPS% panel"

if defined MISSING_DEPS (
    echo [-] Missing dependencies:%MISSING_DEPS%
    echo.
    echo [!] Installing missing dependencies...
    %PYTHON_EXE% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [-] Failed to install dependencies.
        exit /b 1
    )
)
echo     All dependencies installed.

:: Set HF_TOKEN from environment if provided via --token
if defined HF_TOKEN (
    set "HF_TOKEN=%HF_TOKEN%"
)

:: Check CUDA availability
echo [+] Checking GPU/CUDA...
%PYTHON_EXE% -c "import torch; print(torch.cuda.is_available())" 2>nul >temp_cuda.txt
set /p CUDA_AVAILABLE=<temp_cuda.txt
del temp_cuda.txt 2>nul
if "%CUDA_AVAILABLE%"=="True" (
    for /f "tokens=*" %%d in ('%PYTHON_EXE% -c "import torch; print(torch.cuda.get_device_name(0))" 2^>^&1') do set "GPU_NAME=%%d"
    echo     GPU: %GPU_NAME%
) else (
    echo     Running in CPU mode
)

:: Check audio devices
echo [+] Checking audio devices...
%PYTHON_EXE% -c "import sounddevice as sd; print(len(sd.query_devices()))" 2nul >temp_audio.txt
set /p NUM_AUDIO=<temp_audio.txt
del temp_audio.txt 2>nul
if defined NUM_AUDIO (
    echo     Found %NUM_AUDIO% audio device(s)
) else (
    echo     Unable to query audio devices
)

:: Check models if enabled
if "%CHECK_MODELS%"=="1" (
    echo [+] Checking models...
    set "MODELS_OK=1"
    
    :: Check faster-whisper model
    if not exist "%LOCALAPPDATA%\ctranslate2\models" (
        echo     [!] Faster-Whisper model not found
        set "MODELS_OK=0"
    )
    
    if "%NO_DIARIZATION%"=="0" (
        if not defined HF_TOKEN (
            echo     [!] HF_TOKEN required for diarization
            echo     [!] Use --token HF_TOKEN or set HF_TOKEN environment variable
            set "NO_DIARIZATION=1"
        )
    )
    
    if "%MODELS_OK%"=="0" (
        echo.
        echo [!] Models not found. Run download_models.py first?
        echo     python download_models.py --token HF_TOKEN
        echo.
    )
)

:: Build launch command for launcher.py
set "LAUNCH_CMD=%PYTHON_EXE% launcher.py"

if "%NO_GUI%"=="1" set "LAUNCH_CMD=%LAUNCH_CMD% --no-gui"
if "%NO_DIARIZATION%"=="1" set "LAUNCH_CMD=%LAUNCH_CMD% --no-diarization"
set "LAUNCH_CMD=%LAUNCH_CMD% --port %GUI_PORT%"

if "%OPEN_BROWSER%"=="1" set "LAUNCH_CMD=%LAUNCH_CMD% --open-browser"

echo.
echo ============================================================
echo  Starting WhisperTurbo
echo ============================================================
echo.

:: Launch the Python launcher
%LAUNCH_CMD%
exit /b %ERRORLEVEL%

:help
echo.
echo WhisperTurbo Launcher
echo.
echo Usage: launcher.bat [options]
echo.
echo Options:
echo   --no-gui              Disable GUI (run in headless mode)
echo   --no-diarization      Disable speaker diarization
echo   --port PORT           Set GUI port (default: 5006)
echo   --token HF_TOKEN      Set HuggingFace token
echo   --skip-models         Skip model availability check
echo   --no-browser          Don't open browser after startup
echo.
echo Environment Variables:
echo   HF_TOKEN              HuggingFace token for model access
echo.
echo Examples:
echo   launcher.bat                          Start with GUI
echo   launcher.bat --no-gui                 Run headless
echo   launcher.bat --port 8080              Custom port
echo   launcher.bat --token hf_xxx           With token
echo   launcher.bat --no-diarization          No speaker separation
echo.
exit /b 0
