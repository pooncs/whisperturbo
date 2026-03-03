"""
pytest configuration and shared fixtures for WhisperTurbo tests.
"""

import sys
from unittest.mock import MagicMock, patch

mock_pyannote = MagicMock()
sys.modules["pyannote"] = mock_pyannote
sys.modules["pyannote.audio"] = MagicMock()
sys.modules["pyannote.pipeline"] = MagicMock()
sys.modules["pyannote.core"] = MagicMock()

# Mock panel (holoviz) modules
sys.modules["panel"] = MagicMock()
sys.modules["pn"] = MagicMock()  # Common alias
sys.modules["holoviews"] = MagicMock()
sys.modules["bokeh"] = MagicMock()
sys.modules["param"] = MagicMock()

import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset module imports between tests to ensure clean state."""
    yield


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    import numpy as np

    return np.random.rand(16000).astype("float32")


@pytest.fixture
def sample_audio_data_long():
    """Generate longer sample audio data for testing."""
    import numpy as np

    return np.random.rand(16000 * 10).astype("float32")


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    import logging

    with pytest.patch("logging.getLogger") as mock:
        mock_logger = logging.getLogger("test")
        yield mock_logger
