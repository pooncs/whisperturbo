# WhisperTurbo Troubleshooting Guide

## Common Issues and Solutions

### Launcher Issues

#### Launcher Shows Missing Dependencies

**Symptoms**: Running `python launcher.py` shows missing packages.

**Solutions**:
1. Install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. If specific package is missing:
   ```bash
   pip install <package-name>
   ```

3. Use `--skip-models-check` if model check fails incorrectly:
   ```bash
   python launcher.py --skip-models-check
   ```

#### Launcher Browser Doesn't Open

**Symptoms**: GUI starts but browser doesn't open automatically.

**Solutions**:
1. Manually open http://localhost:5006
2. Check if default browser is set
3. Use `--no-browser` flag and open manually
4. Try a different port: `--port 5007`

#### Launcher Shows "Models Not Found"

**Symptoms**: Warning about missing models even after download.

**Solutions**:
1. Download models explicitly:
   ```bash
   python download_models.py
   ```

2. Check model cache location:
   - Windows: `%LOCALAPPDATA%\ctranslate2\models`
   - Linux/macOS: `~/.cache/ctranslate2/models`

3. Use `--skip-models-check` to bypass validation

### Audio Issues

#### No Audio Input

**Symptoms**: Application runs but no audio is captured.

**Solutions**:
1. Check microphone permissions:
   ```bash
   python -c "import sounddevice; print(sounddevice.query_devices())"
   ```

2. Verify default input device is set correctly

3. On Windows, ensure microphone access is enabled in Settings > Privacy > Microphone

#### Audio Latency Too High

**Symptoms**: Noticeable delay between speech and translation.

**Solutions**:
1. Reduce chunk size in config.py
2. Ensure GPU is being used (check CUDA availability)
3. Lower diarization window size (if acceptable quality)

### CUDA/GPU Issues

#### CUDA Not Available

**Symptoms**: `RuntimeError: CUDA not available`

**Solutions**:
1. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Check PyTorch CUDA:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Ensure CUDA 11.8 or 12.x is installed

4. For CUDA 11.8, install PyTorch with:
   ```bash
   pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

5. Check NVIDIA driver version (should be 525+)

#### Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in config
2. Use `int8` compute type instead of `float16`
3. Close other GPU applications
4. Restart the application to clear GPU memory

### Diarization Issues

#### HF_TOKEN Required

**Symptoms**: Diarization fails with authentication error.

**Solutions**:
1. Set HuggingFace token:
   ```bash
   set HF_TOKEN=your_huggingface_token
   ```

2. Get token from: https://huggingface.co/settings/tokens
3. Ensure token has Read permissions
4. Accept terms for pyannote models at:
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0

#### Diarization Very Slow

**Symptoms**: Speaker labels appear with significant delay.

**Solutions**:
1. Reduce diarization window size
2. Use GPU for diarization if possible
3. Consider disabling diarization for faster processing: `--no-diarization`

### Model Issues

#### Model Download Fails

**Symptoms**: Error downloading Faster-Whisper or pyannote models.

**Solutions**:
1. Check internet connection
2. Verify HuggingFace access
3. Try running download_models.py separately:
   ```bash
   python download_models.py
   ```

#### Invalid Model Name

**Symptoms**: Model loading fails.

**Solutions**:
1. Verify model name in config.py
2. Check for typos in model identifier
3. Ensure model exists on HuggingFace Hub

### GUI Issues

#### GUI Not Loading

**Symptoms**: Cannot access http://localhost:5006

**Solutions**:
1. Check if port is in use:
   ```bash
   netstat -ano | findstr :5006
   ```

2. Try different port: `--gui-port 5007`

3. Check browser console for errors

4. Run in headless mode to test core functionality:
   ```bash
   python main.py --no-gui
   ```

#### GUI Not Updating

**Symptoms**: Table doesn't show new translations.

**Solutions**:
1. Check browser for JavaScript errors
2. Increase GUI refresh rate in config
3. Check network tab for WebSocket issues

### Export Issues

#### Export File Empty

**Symptoms**: Exported CSV/JSONL/SRT files are empty.

**Solutions**:
1. Wait for more translations to accumulate
2. Check file permissions
3. Verify export path is writable

### Performance Issues

#### High CPU Usage

**Symptoms**: System becomes sluggish.

**Solutions**:
1. Use GPU acceleration
2. Reduce audio chunk size
3. Limit maximum table rows

#### Low Translation Quality

**Symptoms**: Translations are inaccurate.

**Solutions**:
1. Ensure source language is set correctly (ko for Korean)
2. Use larger Whisper model if available
3. Check audio input quality
4. Minimize background noise
5. Enable context carry for better continuity

### Code Quality Checks

#### Ruff Linting

To check code quality with ruff:

```bash
# Check all files
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Check specific file
ruff check src/whisper_asr.py
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_whisper_asr.py

# Run with coverage
pytest --cov=src
```

## Debug Mode

Enable debug logging for more detailed error information:

```bash
python main.py --log-level DEBUG
```

Or with the launcher:

```bash
python launcher.py --log-level DEBUG
```

## Benchmark Mode

For performance diagnostics, use benchmark mode:

```bash
python main.py --benchmark
```

This will display:
- Per-cycle latency and RTF metrics
- Aggregate statistics every 5 cycles
- Total segment counts

## Getting Help

If you encounter issues not listed here:

1. Check the main README.md
2. Review ARCHITECTURE.md for system design
3. Check CONFIGURATION.md for settings
4. Enable DEBUG logging for detailed diagnostics
5. Run launcher.py for environment validation
