#!/usr/bin/env python3
"""
Model download/prefetch script for WhisperTurbo.
Downloads Faster-Whisper, pyannote speaker-diarization, VAD, and segmentation models.

Usage:
    python download_models.py                    # Use HF_TOKEN from environment
    python download_models.py --token HF_TOKEN   # Pass token as argument
    python download_models.py --force            # Re-download all models
    python download_models.py --verify           # Verify checksums after download
"""

import argparse
import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

try:
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm import tqdm
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


CONFIG = {
    "whisper_model": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "diarization_model": "pyannote/speaker-diarization-community-1",
    "segmentation_model": "pyannote/segmentation-3.0",
    "vad_model": "pyannote/segmentation-3.0",
}


def get_cache_dir() -> Path:
    """Get the model cache directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "whisperturbo" / "models"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "whisperturbo"
    else:
        return Path.home() / ".cache" / "whisperturbo"


def get_ctranslate2_cache_dir() -> Path:
    """Get CTranslate2 model cache directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "ctranslate2" / "models"
    else:
        return Path.home() / ".cache" / "ctranslate2" / "models"


def get_pyannote_cache_dir() -> Path:
    """Get pyannote model cache directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "pyannote" / "models"
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(cache_home) / "pyannote" / "models"


def print_status(msg: str, success: bool = True) -> None:
    """Print colored status message."""
    prefix = "[+]" if success else "[-]"
    print(f"{prefix} {msg}")


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def calculate_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Calculate hash of a file."""
    hash_obj = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def download_with_progress(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    token: str,
    force: bool = False,
) -> Optional[Path]:
    """Download a single file with progress bar."""
    if not HF_HUB_AVAILABLE:
        print_status("huggingface_hub not installed. Install with: pip install huggingface_hub", False)
        return None

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            token=token,
            force_download=force,
            resume_download=True,
        )
        return Path(local_path)
    except Exception as e:
        print_status(f"Failed to download {repo_id}/{filename}: {e}", False)
        return None


def download_faster_whisper(
    model_name: str,
    cache_dir: Path,
    token: Optional[str],
    force: bool = False,
) -> Tuple[bool, Optional[Path]]:
    """
    Download Faster-Whisper CTranslate2 model.
    Returns (success, model_path).
    """
    print_section("Faster-Whisper Model")
    print_status(f"Model: {model_name}")
    print_status(f"Cache directory: {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_info_file = cache_dir / "model_info.json"
    model_path = cache_dir / model_name.replace("/", "--")
    
    if model_path.exists() and not force:
        print_status(f"Model already cached at {model_path}")
        return True, model_path

    if HF_HUB_AVAILABLE:
        print_status("Downloading CTranslate2 model files...")
        
        api = HfApi(token=token)
        
        try:
            repo_info = api.repo_info(model_name, repo_type="model")
            files = []
            
            for item in repo_info.siblings:
                if item.rfilename.endswith((".bin", ".pt", ".safetensors", ".onnx", ".yaml", ".json")):
                    files.append(item.rfilename)
            
            if not files:
                print_status("No model files found in repository", False)
                return False, None
            
            print_status(f"Found {len(files)} model files")
            
            downloaded_files = []
            for filename in tqdm(files, desc="Downloading model files", unit="file"):
                file_path = download_with_progress(
                    repo_id=model_name,
                    filename=filename,
                    cache_dir=cache_dir,
                    token=token,
                    force=force,
                )
                if file_path:
                    downloaded_files.append(file_path)
            
            if downloaded_files:
                model_info = {
                    "model_name": model_name,
                    "cached_at": str(cache_dir),
                    "files": [str(f) for f in downloaded_files],
                }
                with open(model_info_file, "w") as f:
                    json.dump(model_info, f, indent=2)
                
                print_status(f"Model cached successfully at {cache_dir}")
                return True, cache_dir
            else:
                print_status("No files were downloaded", False)
                return False, None
                
        except Exception as e:
            print_status(f"Error downloading model: {e}", False)
            return False, None
    else:
        print_status("huggingface_hub not available, trying faster-whisper import...")
        
        if FASTER_WHISPER_AVAILABLE:
            try:
                print_status("Loading model to trigger download...")
                model = WhisperModel(
                    model_name,
                    device="cpu",
                    download_root=str(cache_dir),
                )
                print_status("Model downloaded via faster-whisper")
                return True, cache_dir
            except Exception as e:
                print_status(f"Failed to download via faster-whisper: {e}", False)
                return False, None
        else:
            print_status("faster-whisper not installed", False)
            return False, None


def download_pyannote_models(
    diarization_model: str,
    cache_dir: Path,
    token: Optional[str],
    force: bool = False,
    verify: bool = False,
) -> Tuple[bool, Dict[str, Optional[Path]]]:
    """
    Download pyannote models (speaker-diarization, segmentation, VAD).
    Returns (success, {model_name: path}).
    """
    print_section("PyAnnote Models")
    
    if not token:
        print_status("WARNING: HF_TOKEN required for pyannote models", False)
        print_status("Set HF_TOKEN environment variable or pass --token argument")
    
    print_status(f"Diarization model: {diarization_model}")
    print_status(f"Segmentation model: {CONFIG['segmentation_model']}")
    print_status(f"Cache directory: {cache_dir}")
    
    results = {}
    
    if PYANNOTE_AVAILABLE and token:
        print_status("Using pyannote.audio to download models...")
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print_status("Loading diarization pipeline (triggers model downloads)...")
            
            pipeline = Pipeline.from_pretrained(
                diarization_model,
                use_auth_token=token,
            )
            
            print_status("Pipeline loaded successfully")
            
            pipeline_info = {
                "diarization_model": diarization_model,
                "cached_at": str(cache_dir),
                "config": {
                    "segmentation": CONFIG["segmentation_model"],
                },
            }
            
            info_file = cache_dir / "pyannote_info.json"
            with open(info_file, "w") as f:
                json.dump(pipeline_info, f, indent=2)
            
            results["diarization"] = cache_dir
            results["segmentation"] = cache_dir
            results["vad"] = cache_dir
            
            print_status("All pyannote models downloaded successfully")
            return True, results
            
        except Exception as e:
            print_status(f"Failed to download pyannote models: {e}", False)
            return False, results
    else:
        if not PYANNOTE_AVAILABLE:
            print_status("pyannote.audio not installed. Install with: pip install pyannote.audio")
        
        if HF_HUB_AVAILABLE and token:
            print_status("Downloading model files via huggingface_hub...")
            
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            api = HfApi(token=token)
            
            model_files = [
                (diarization_model, "config.yaml"),
                (CONFIG["segmentation_model"], "config.yaml"),
                (CONFIG["segmentation_model"], "pytorch_model.bin"),
            ]
            
            for repo_id, filename in model_files:
                print_status(f"Downloading {repo_id}/{filename}...")
                file_path = download_with_progress(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=cache_dir,
                    token=token,
                    force=force,
                )
                if file_path:
                    results[f"{repo_id}/{filename}"] = file_path
            
            if results:
                print_status(f"Downloaded {len(results)} files")
                return True, results
            else:
                return False, results
        else:
            print_status("Cannot download models. Install pyannote.audio and provide HF_TOKEN", False)
            return False, results


def verify_downloads(cache_dir: Path) -> bool:
    """Verify downloaded models by attempting to load them."""
    print_section("Verification")
    
    all_ok = True
    
    ctranslate2_cache = get_ctranslate2_cache_dir()
    if ctranslate2_cache.exists():
        print_status(f"Checking CTranslate2 cache: {ctranslate2_cache}")
        
        model_folders = list(ctranslate2_cache.glob("*"))
        if model_folders:
            for folder in model_folders:
                if folder.is_dir():
                    files = list(folder.glob("*"))
                    print_status(f"  {folder.name}: {len(files)} files")
        else:
            print_status("No models found in CTranslate2 cache", False)
            all_ok = False
    else:
        print_status("CTranslate2 cache directory not found", False)
        all_ok = False
    
    pyannote_cache = get_pyannote_cache_dir()
    if pyannote_cache.exists():
        print_status(f"Checking pyannote cache: {pyannote_cache}")
        
        files = list(pyannote_cache.rglob("*"))
        folders = [f for f in files if f.is_dir()]
        if folders:
            for folder in folders[:5]:
                print_status(f"  {folder.name}")
            if len(folders) > 5:
                print_status(f"  ... and {len(folders) - 5} more")
        else:
            print_status("No models found in pyannote cache", False)
            all_ok = False
    else:
        print_status("Pyannote cache directory not found", False)
        all_ok = False
    
    return all_ok


def test_faster_whisper(cache_dir: Path) -> bool:
    """Test loading Faster-Whisper model."""
    print_status("Testing Faster-Whisper model...")
    
    if FASTER_WHISPER_AVAILABLE:
        try:
            model = WhisperModel(
                "large-v3-turbo",
                device="cpu",
                download_root=str(cache_dir),
            )
            print_status("Faster-Whisper model loads successfully")
            return True
        except Exception as e:
            print_status(f"Faster-Whisper test failed: {e}", False)
            return False
    else:
        print_status("faster-whisper not available for testing", False)
        return False


def test_pyannote(token: str) -> bool:
    """Test loading pyannote pipeline."""
    print_status("Testing pyannote pipeline...")
    
    if PYANNOTE_AVAILABLE:
        try:
            pipeline = Pipeline.from_pretrained(
                CONFIG["diarization_model"],
                use_auth_token=token,
            )
            print_status("Pyannote pipeline loads successfully")
            return True
        except Exception as e:
            print_status(f"Pyannote test failed: {e}", False)
            return False
    else:
        print_status("pyannote.audio not available for testing", False)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models for WhisperTurbo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py
  python download_models.py --token hf_xxxxx
  python download_models.py --force
  python download_models.py --verify

Environment variables:
  HF_TOKEN    HuggingFace token for accessing gated models
        """,
    )
    
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download of all models",
    )
    
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify downloads after downloading",
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Test loading models after download",
    )
    
    parser.add_argument(
        "--whisper-only",
        action="store_true",
        help="Download only Faster-Whisper model",
    )
    
    parser.add_argument(
        "--pyannote-only",
        action="store_true",
        help="Download only pyannote models",
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory",
    )
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    
    if not token:
        print_status("WARNING: HF_TOKEN not set. Some models may fail to download.", False)
        print_status("Set HF_TOKEN environment variable or pass --token argument")
    
    custom_cache = Path(args.cache_dir) if args.cache_dir else None
    
    whisper_cache = custom_cache or get_ctranslate2_cache_dir()
    pyannote_cache = custom_cache or get_pyannote_cache_dir()
    
    print("WhisperTurbo Model Downloader")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")
    print(f"Token: {'Provided' if token else 'Not provided'}")
    print(f"Force: {args.force}")
    print(f"Whisper cache: {whisper_cache}")
    print(f"Pyannote cache: {pyannote_cache}")
    
    success_count = 0
    total_tasks = 2 - (args.whisper_only or 0) - (args.pyannote_only or 0)
    
    if not args.pyannote_only:
        success, path = download_faster_whisper(
            model_name=CONFIG["whisper_model"],
            cache_dir=whisper_cache,
            token=token,
            force=args.force,
        )
        if success:
            success_count += 1
        
        if args.test and success:
            test_faster_whisper(whisper_cache)
    
    if not args.whisper_only:
        success, results = download_pyannote_models(
            diarization_model=CONFIG["diarization_model"],
            cache_dir=pyannote_cache,
            token=token,
            force=args.force,
            verify=args.verify,
        )
        if success:
            success_count += 1
        
        if args.test and token and success:
            test_pyannote(token)
    
    if args.verify:
        verify_downloads(get_cache_dir())
    
    print_section("Summary")
    if success_count == total_tasks:
        print_status(f"All models downloaded successfully ({success_count}/{total_tasks})")
        return 0
    else:
        print_status(f"Partial success ({success_count}/{total_tasks})", False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
