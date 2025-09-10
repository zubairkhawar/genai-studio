#!/usr/bin/env python3
"""
Generate voice preview samples for each voice style
"""

import os
import sys
import asyncio
import pathlib
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from models.audio_generator import AudioGenerator

# Sample text for voice previews
SAMPLE_TEXT = "Hello! This is a preview of how this voice style sounds. You can use this to help choose the perfect voice for your audio generation."

# Voice styles and their preview texts
VOICE_PREVIEWS = {
    "auto": "Hello! This is an automatically selected voice that adapts to your content.",
    "professional": "Good day. This is a professional voice suitable for business presentations and formal communications.",
    "casual": "Hey there! This is a casual, friendly voice perfect for everyday conversations and informal content.",
    "storytelling": "Once upon a time, in a land far away, there lived a storyteller with a voice that captivated all who listened.",
    "character": "Greetings, mortal! I am a character voice with personality and flair, perfect for animated content and creative projects.",
    "male": "Hello, this is a deep, masculine voice that conveys authority and confidence in your audio content.",
    "female": "Hi there! This is a clear, feminine voice that's warm and engaging for all types of audio content.",
    "child": "Hi! I'm a young, energetic voice that's perfect for children's content and fun, playful audio!",
    "narrator": "Welcome. This is a narrator voice, clear and authoritative, ideal for documentaries and educational content."
}

async def generate_voice_previews():
    """Generate preview audio files for each voice style"""
    
    # Create previews directory
    previews_dir = Path("../outputs/voice-previews")
    previews_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize audio generator with GPU info
    from utils.gpu_detector import GPUDetector
    gpu_detector = GPUDetector()
    gpu_info = gpu_detector.detect_gpu()
    audio_generator = AudioGenerator(gpu_info)
    
    print("🎤 Generating voice preview samples...")
    print(f"📁 Output directory: {previews_dir}")
    print()
    
    # Generate preview for each voice style
    for voice_style, preview_text in VOICE_PREVIEWS.items():
        try:
            print(f"🎵 Generating preview for '{voice_style}' voice...")
            
            # Generate audio with specific voice style
            output_path = await audio_generator.generate(
                prompt=preview_text,
                model_name="bark",
                sample_rate=22050,
                output_format="wav",
                voice_style=voice_style
            )
            
            # Copy to previews directory with standardized name
            import shutil
            preview_filename = f"{voice_style}-preview.wav"
            preview_path = previews_dir / preview_filename
            
            shutil.copy2(output_path, preview_path)
            
            print(f"✅ Generated: {preview_filename}")
            print(f"   Text: {preview_text[:50]}...")
            print()
            
        except Exception as e:
            print(f"❌ Failed to generate preview for '{voice_style}': {e}")
            print()
    
    print("🎉 Voice preview generation complete!")
    print(f"📂 Preview files saved to: {previews_dir}")
    
    # List generated files
    print("\n📋 Generated preview files:")
    for file in sorted(previews_dir.glob("*.wav")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   • {file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    asyncio.run(generate_voice_previews())
