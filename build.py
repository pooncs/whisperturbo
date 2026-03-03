#!/usr/bin/env python3
"""
Build script for WhisperTurbo Windows executable
Run: python build.py
Run with lint: python build.py lint
"""

import os
import sys
import shutil
import subprocess


def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller

        return True
    except ImportError:
        return False


def run_lint():
    """Run ruff linter on the project"""
    print("Running ruff linter...")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/", "tests/", "main.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return result.returncode == 0


def clean_build_dirs():
    """Clean previous build directories"""
    dirs_to_clean = ["dist", "build", "__pycache__"]

    for dirname in dirs_to_clean:
        if os.path.exists(dirname):
            print(f"Cleaning {dirname}/")
            shutil.rmtree(dirname, ignore_errors=True)

    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            pycache = os.path.join(root, "__pycache__")
            print(f"Cleaning {pycache}/")
            shutil.rmtree(pycache, ignore_errors=True)


def run_pyinstaller():
    """Run PyInstaller with the spec file or generate spec if needed"""
    spec_file = "whisperturbo.spec"

    if not os.path.exists(spec_file):
        print(f"WARNING: {spec_file} not found!")
        print("Generating spec file automatically...")

        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--name=WhisperTurbo",
            "--onefile",
            "--console=False",
            "--add-data=src;src",
            "--hidden-import=faster_whisper",
            "--hidden-import=torch",
            "--hidden-import=torch.cuda",
            "--hidden-import=pyannote",
            "--hidden-import=pyannote.audio",
            "--hidden-import=silero_vad",
            "--hidden-import=panel",
            "--hidden-import=sounddevice",
            "--runtime-hook=runtime_hook.py",
            "--exclude-module=tests",
            "--clean",
            "--noconfirm",
            "main.py",
        ]

        print(f"Command: {' '.join(cmd)}")
        print()

        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode == 0

    cmd = [sys.executable, "-m", "PyInstaller", spec_file, "--clean", "--noconfirm"]

    print("Running PyInstaller...")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    print("=" * 40)
    print("WhisperTurbo Build Script")
    print("=" * 40)
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "lint":
        if not run_lint():
            sys.exit(1)
        return

    if not check_pyinstaller():
        print("ERROR: PyInstaller not found.")
        print("Install with: pip install pyinstaller")
        sys.exit(1)

    print("[1/3] Cleaning previous builds...")
    clean_build_dirs()
    print()

    print("[2/3] Running PyInstaller...")
    if not run_pyinstaller():
        print("\nERROR: PyInstaller failed!")
        sys.exit(1)
    print()

    print("[3/3] Build complete!")
    print()
    print("=" * 40)
    print("Output: dist/WhisperTurbo/")
    print("=" * 40)
    print()
    print("NOTE: For RTX 4090 CUDA support, ensure:")
    print("  1. NVIDIA Driver 535+ is installed")
    print("  2. CUDA Toolkit 12.x is installed")
    print("  3. cuDNN 8.x is installed")
    print("  4. HF_TOKEN environment variable is set")
    print()
    print("Run the application:")
    print("  dist\\WhisperTurbo\\WhisperTurbo.exe")
    print()


if __name__ == "__main__":
    main()
