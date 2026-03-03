#!/usr/bin/env python3
"""
Build script for WhisperTurbo Windows executable
Run: python build.py
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
    """Run PyInstaller with the spec file"""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "whisperturbo.spec",
        "--clean",
        "--noconfirm"
    ]
    
    print("Running PyInstaller...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    print("=" * 40)
    print("WhisperTurbo Windows Build Script")
    print("=" * 40)
    print()
    
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
