@echo off
REM Build script for WhisperTurbo Windows executable
REM Requires PyInstaller: pip install pyinstaller

echo ========================================
echo WhisperTurbo Windows Build Script
echo ========================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo ERROR: PyInstaller not found. Install with:
    echo   pip install pyinstaller
    exit /b 1
)

echo [1/3] Cleaning previous builds...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "__pycache__" rmdir /s /q "__pycache__"
for /d %%d in (__pycache__*) do rmdir /s /q "%%d"

echo [2/3] Running PyInstaller...
echo.
pyinstaller whisperturbo.spec --clean --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller failed!
    exit /b 1
)

echo [3/3] Build complete!
echo.
echo ========================================
echo Output: dist\WhisperTurbo\
echo ========================================
echo.
echo NOTE: For RTX 4090 CUDA support, ensure:
echo   1. NVIDIA Driver 535+ is installed
echo   2. CUDA Toolkit 12.x is installed
echo   3. cuDNN 8.x is installed
echo   4. HF_TOKEN environment variable is set
echo.
echo Run the application:
echo   dist\WhisperTurbo\WhisperTurbo.exe
echo.

pause
