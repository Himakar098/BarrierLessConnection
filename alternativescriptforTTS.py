import os
import argparse
import torch

# Test if basic imports work
print("Testing imports...")
import numpy as np
print("✓ NumPy")
import librosa
print("✓ Librosa")
import soundfile as sf
print("✓ SoundFile")
from transformers import pipeline
print("✓ Transformers")

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test TTS with safer options
print("\nTesting TTS initialization...")
try:
    # Add safe globals for PyTorch 2.6+
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        try:
            torch.serialization.add_safe_globals([XttsConfig])
            print("✓ Added XttsConfig to safe globals")
        except AttributeError:
            # For older PyTorch versions
            print("⚠ PyTorch version doesn't have add_safe_globals method")
    except ImportError:
        print("⚠ Could not import XttsConfig")
    
    # Try to initialize TTS with weights_only=False
    print("\nAttempting TTS initialization with alternative loading method...")
    # Override torch.load temporarily to use weights_only=False
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    # Apply the patch and try loading
    torch.load = patched_torch_load
    
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")  # Try a simpler model first
        print("✓ Successfully initialized TTS with Tacotron2")
    except Exception as e:
        print(f"✗ Failed to initialize Tacotron2: {str(e)}")
        try:
            # Try an even simpler model
            tts = TTS("tts_models/en/ljspeech/glow-tts")
            print("✓ Successfully initialized TTS with Glow-TTS")
        except Exception as e:
            print(f"✗ Failed to initialize Glow-TTS: {str(e)}")
            print("Trying an alternative approach...")
    
    # Restore original torch.load
    torch.load = original_torch_load
    
except Exception as e:
    print(f"✗ TTS test failed: {str(e)}")

# Test gTTS as fallback
print("\nTesting gTTS as fallback...")
try:
    from gtts import gTTS
    print("✓ gTTS available as fallback")
    
    # Test a simple tts generation
    output_file = "gtts_test.mp3"
    tts = gTTS(text="This is a test of the fallback TTS system.", lang="en")
    tts.save(output_file)
    print(f"✓ Sample gTTS file created: {output_file}")
except Exception as e:
    print(f"✗ gTTS test failed: {str(e)}")

print("\nTest complete. Check the issues above and use the appropriate TTS implementation.")