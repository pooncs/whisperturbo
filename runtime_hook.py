"""
Runtime hook for WhisperTurbo cleanup on exit
Ensures proper cleanup of CUDA contexts, audio streams, and panel server
"""

import os
import sys
import logging
import atexit
import gc

logger = logging.getLogger(__name__)


def cleanup_cuda():
    """Cleanup CUDA resources"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("CUDA cache cleared")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"CUDA cleanup error: {e}")


def cleanup_torch():
    """Cleanup torch multiprocessing resources"""
    try:
        import torch
        if hasattr(torch, 'multiprocessing'):
            torch.multiprocessing.shutdown()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Torch cleanup error: {e}")


def cleanup_audio():
    """Cleanup audio input resources"""
    try:
        import sounddevice as sd
        if hasattr(sd, 'stop'):
            sd.stop()
        if hasattr(sd, 'close'):
            sd.close()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Audio cleanup error: {e}")


def cleanup_panel():
    """Cleanup Panel/Bokeh server resources"""
    try:
        import panel
        if hasattr(panel, 'cleanup'):
            panel.cleanup()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Panel cleanup error: {e}")


def cleanup_threading():
    """Ensure all threads are properly terminated"""
    import threading
    
    current_thread = threading.current_thread()
    
    for thread in threading.enumerate():
        if thread != current_thread and thread.daemon:
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass


def cleanup_logging():
    """Flush and close logging handlers"""
    try:
        logging.shutdown()
    except Exception:
        pass


def force_garbage_collection():
    """Force garbage collection to release resources"""
    try:
        gc.collect()
    except Exception:
        pass


def on_exit():
    """Main cleanup function called on exit"""
    logger.info("Running application cleanup...")
    
    cleanup_panel()
    cleanup_audio()
    cleanup_cuda()
    cleanup_torch()
    cleanup_threading()
    cleanup_logging()
    force_garbage_collection()
    
    logger.info("Application cleanup completed")


atexit.register(on_exit)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    import signal
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        on_exit()
        sys.exit(0)
    
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)


setup_signal_handlers()
