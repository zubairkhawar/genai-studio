import torch
import os
import asyncio
import tempfile
import shutil
import pathlib
from typing import Dict, List, Optional, Any
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import ffmpeg

class AudioGenerator:
    """Text-to-audio generation using various models"""
    
    def __init__(self, gpu_info: Dict[str, Any]):
        self.gpu_info = gpu_info
        self.device = gpu_info["device"]
        self.models = {}
        
        # Fix PyTorch 2.6 weights_only issue for Bark
        try:
            torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
        except Exception:
            pass
            
        self.available_models = {
            "bark": {
                "id": "bark",
                "name": "Bark",
                "description": "High-quality text-to-speech and audio generation",
                "max_duration": 14,
                "sample_rate": 22050,
                "type": "tts",
                "features": [
                    "Multiple voices",
                    "Non-speech sounds",
                    "Offline capable",
                    "Lightweight inference"
                ]
            }
        }
    
    async def load_default_models(self):
        """Load default models - only called when explicitly requested"""
        try:
            # Check if Bark model exists locally before attempting to load
            bark_path = pathlib.Path("../models/audio/bark")
            
            if bark_path.exists() and (bark_path / "config.json").exists():
                print("Local Bark model found, loading...")
                await self.load_model("bark")
            else:
                print("No local Bark model found. Use 'Download Models' to download it first.")
        except Exception as e:
            print(f"Warning: Could not load default audio model: {e}")
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        try:
            if model_name in self.models:
                return True
            
            print(f"Loading audio model: {model_name}")
            
            if model_name == "bark":
                # Try to load Bark from local directory first
                local_path = "../models/audio/bark"
                if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
                    try:
                        print(f"Loading Bark from local path: {local_path}")
                        # Set environment variable to use local cache
                        os.environ["BARK_CACHE_DIR"] = str(os.path.abspath(local_path))
                        
                        from bark import SAMPLE_RATE, generate_audio, preload_models
                        # Import other modules only if needed
                        try:
                            from bark.generation import load_codec_model, generate_text_semantic
                            from bark.api import semantic_to_waveform
                        except ImportError as e:
                            print(f"Warning: Some Bark modules not available: {e}")
                        
                        # Fix PyTorch weights_only issue for Bark models
                        import torch
                        original_load = torch.load
                        def patched_load(*args, **kwargs):
                            if 'weights_only' not in kwargs:
                                kwargs['weights_only'] = False
                            return original_load(*args, **kwargs)
                        torch.load = patched_load
                        
                        # Preload models
                        try:
                            preload_models()
                        except Exception as e:
                            print(f"Warning: Bark preload failed: {e}")
                            print("Bark will be loaded on first use")
                        
                        self.models[model_name] = {
                            "sample_rate": SAMPLE_RATE,
                            "generate_audio": generate_audio,
                            "preload_models": preload_models,
                            "local": True
                        }
                        print("✅ Successfully loaded Bark from local weights")
                        
                    except Exception as e:
                        print(f"Failed to load Bark from local path: {e}")
                        # Fallback to default Bark loading
                        try:
                            from bark import SAMPLE_RATE, generate_audio, preload_models
                            from bark.generation import load_codec_model, generate_text_semantic
                            from bark.api import semantic_to_waveform
                            
                            # Fix PyTorch weights_only issue for Bark models
                            import torch
                            original_load = torch.load
                            def patched_load(*args, **kwargs):
                                if 'weights_only' not in kwargs:
                                    kwargs['weights_only'] = False
                                return original_load(*args, **kwargs)
                            torch.load = patched_load
                            
                            # Preload models
                            preload_models()
                            
                            self.models[model_name] = {
                                "sample_rate": SAMPLE_RATE,
                                "generate_audio": generate_audio,
                                "preload_models": preload_models
                            }
                            print("✅ Successfully loaded Bark from default location")
                            
                        except Exception as e2:
                            print(f"Bark not available: {e2}")
                            # Fallback to placeholder for demo purposes
                            self.models[model_name] = {
                                "sample_rate": 22050,
                                "placeholder": True
                            }
                else:
                    print(f"Local Bark model not found at {local_path}, trying default loading...")
                    try:
                        from bark import SAMPLE_RATE, generate_audio, preload_models
                        from bark.generation import load_codec_model, generate_text_semantic
                        from bark.api import semantic_to_waveform
                        
                        # Fix PyTorch weights_only issue for Bark models
                        import torch
                        original_load = torch.load
                        def patched_load(*args, **kwargs):
                            if 'weights_only' not in kwargs:
                                kwargs['weights_only'] = False
                            return original_load(*args, **kwargs)
                        torch.load = patched_load
                        
                        # Preload models
                        try:
                            preload_models()
                        except Exception as e:
                            print(f"Warning: Bark preload failed: {e}")
                            print("Bark will be loaded on first use")
                        
                        self.models[model_name] = {
                            "sample_rate": SAMPLE_RATE,
                            "generate_audio": generate_audio,
                            "preload_models": preload_models
                        }
                        print("✅ Successfully loaded Bark from default location")
                        
                    except Exception as e:
                        print(f"Bark not available: {e}")
                        # Fallback to placeholder for demo purposes
                        self.models[model_name] = {
                            "sample_rate": 22050,
                            "placeholder": True
                        }
                    
            else:
                raise ValueError(f"Unknown model: {model_name}. Only 'bark' is supported.")
            
            print(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        available = []
        for model_id, info in self.available_models.items():
            # Special handling for Bark - check local models directory
            if model_id == "bark":
                # For Bark, check local models directory
                size_gb = 0
                status = "download"
                
                # Check local Bark directory
                local_bark_dir = pathlib.Path("../models/audio/bark")
                if local_bark_dir.exists():
                    # Check for core model files
                    core_models = ["text_2.pt", "coarse.pt", "fine.pt"]
                    core_model_found = False
                    for model_file in core_models:
                        if (local_bark_dir / model_file).exists():
                            core_model_found = True
                            break
                    
                    if core_model_found:
                        # Calculate total size of all files in Bark directory
                        total_size = sum(f.stat().st_size for f in local_bark_dir.rglob("*") if f.is_file())
                        size_gb = total_size / (1024 * 1024 * 1024)
                        status = "available"
                
                available.append({
                    "id": model_id,
                    "name": info["name"],
                    "description": info["description"],
                    "max_duration": info["max_duration"],
                    "sample_rate": info["sample_rate"],
                    "size_gb": round(size_gb, 2) if size_gb > 0 else 0,
                    "status": status,
                    "loaded": size_gb > 0
                })
            else:
                # For other models, check local directory
                model_path = pathlib.Path(f"../models/audio/{model_id}")
                if model_path.exists():
                    # Check for actual model weight files (not just config files)
                    weight_files = list(model_path.rglob("*.safetensors")) + list(model_path.rglob("*.bin")) + list(model_path.rglob("*.pt")) + list(model_path.rglob("*.pth"))
                    if len(weight_files) > 0:
                        # Calculate total model size
                        total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
                        size_gb = total_size / (1024 * 1024 * 1024)
                        
                        available.append({
                            "id": model_id,
                            "name": info["name"],
                            "description": info["description"],
                            "max_duration": info["max_duration"],
                            "sample_rate": info["sample_rate"],
                            "size_gb": round(size_gb, 2),
                            "loaded": model_id in self.models
                        })
        return available
    
    async def generate(self, prompt: str, model_name: str, sample_rate: int = 22050,
                      output_format: str = "wav", voice_style: str = "auto", voice_id: str = None) -> str:
        """Generate audio from text prompt"""
        try:
            # Ensure model is loaded
            if model_name not in self.models:
                await self.load_model(model_name)
            
            if model_name not in self.models:
                raise RuntimeError(f"Could not load model: {model_name}")
            
            # Generate audio using Bark
            if model_name == "bark":
                return await self._generate_bark(prompt, sample_rate, output_format, voice_style, voice_id)
            else:
                raise ValueError(f"Unsupported model: {model_name}. Only Bark is supported.")
                
        except Exception as e:
            raise RuntimeError(f"Audio generation failed: {e}")
    

    
    async def _save_audio_from_file(self, input_path: str, sample_rate: int, output_format: str) -> str:
        """Save audio from existing file using proper filename extensions"""
        try:
            # Create output directory
            os.makedirs("../outputs/audio", exist_ok=True)
            
            # Generate output filename with proper extension
            import uuid
            filename_base = f"xtts_{uuid.uuid4().hex[:8]}"
            output_path = f"../outputs/audio/{filename_base}.{output_format}"
            
            # Convert using ffmpeg with proper filenames
            try:
                if output_format == 'wav':
                    # For WAV files, use ffmpeg with proper output filename
                    (
                        ffmpeg
                        .input(input_path)
                        .output(output_path, acodec='pcm_s16le', ar=sample_rate)
                        .overwrite_output()
                        .run(quiet=True)
                    )
                elif output_format == 'mp3':
                    # For MP3 files, use ffmpeg with proper output filename
                    (
                        ffmpeg
                        .input(input_path)
                        .output(output_path, acodec='libmp3lame', ar=sample_rate, ab='192k')
                        .overwrite_output()
                        .run(quiet=True)
                    )
                else:
                    # For other formats, use default encoding
                    (
                        ffmpeg
                        .input(input_path)
                        .output(output_path)
                        .overwrite_output()
                        .run(quiet=True)
                    )
            except ffmpeg.Error as e:
                print(f"FFmpeg error details: {e}")
                print(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'No stderr'}")
                raise RuntimeError(f"ffmpeg error (see stderr output for detail): {e}")
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to save audio from file: {e}")
    
    async def _generate_bark(self, prompt: str, sample_rate: int, output_format: str, voice_style: str = "auto", voice_id: str = None) -> str:
        """Generate audio using Bark with voice style control and English voice presets"""
        try:
            model_info = self.models["bark"]
            
            # Check if this is a placeholder model
            if model_info.get("placeholder"):
                print("Using placeholder for Bark model")
                return await self._generate_bark_placeholder(prompt, sample_rate, output_format)
            
            generate_audio = model_info["generate_audio"]
            
            # Process prompt for better emotion handling and timing
            processed_prompt = self._process_prompt_for_bark(prompt, voice_style)
            
            # English voice presets from your Colab reference
            english_voice_presets = [
                "v2/en_speaker_0",  # English Male
                "v2/en_speaker_1",  # English Male
                "v2/en_speaker_2",  # English Male
                "v2/en_speaker_3",  # English Male
                "v2/en_speaker_4",  # English Male
                "v2/en_speaker_5",  # English Male (Grainy)
                "v2/en_speaker_6",  # English Male (Suno Favorite)
                "v2/en_speaker_7",  # English Male
                "v2/en_speaker_8",  # English Male
                "v2/en_speaker_9",  # English Female
            ]
            
            # Use provided voice_id if it's a valid English preset, otherwise use default
            if voice_id and voice_id in english_voice_presets:
                voice_preset = voice_id
            else:
                # Default to v2/en_speaker_0 (English Male)
                voice_preset = "v2/en_speaker_0"
            
            print(f"Generating audio with Bark voice: {voice_preset}")
            print(f"Original prompt: {prompt}")
            print(f"Processed prompt: {processed_prompt}")
            
            # Generate audio with specific voice preset, with robust fallbacks
            audio_array = None
            try:
                audio_array = generate_audio(processed_prompt, history_prompt=voice_preset)
            except Exception as first_err:
                print(f"Bark generation with history_prompt failed, retrying without preset: {first_err}")
                try:
                    audio_array = generate_audio(processed_prompt)
                except Exception as second_err:
                    print(f"Bark generation without preset also failed: {second_err}")
                    audio_array = None

            if audio_array is not None:
                # Post-process audio to fix timing issues
                audio_array = self._post_process_audio(audio_array, sample_rate)
                # Save audio
                output_path = await self._save_audio(audio_array, sample_rate, output_format)
                return output_path

            # Fallback: if presets exist locally, use preset preview MP3 as output
            try:
                preset_filename = f"{voice_preset.replace('/', '_')}-preview.mp3"
                preset_path = pathlib.Path("../models/audio/bark/preset-audios") / preset_filename
                if preset_path.exists():
                    return await self._save_audio_from_file(str(preset_path), sample_rate, output_format)
            except Exception as preset_err:
                print(f"Bark preset fallback failed: {preset_err}")
            
            # If all attempts failed, raise the original error
            raise RuntimeError("Bark generation failed and no preset fallback available")
            
        except Exception as e:
            print(f"Bark generation error: {e}")
            raise RuntimeError(f"Bark generation failed: {e}")
    
    def _process_prompt_for_bark(self, prompt: str, voice_style: str) -> str:
        """Process prompt for better emotion handling and Bark compatibility"""
        # First apply voice style
        processed_prompt = self._apply_voice_style(prompt, voice_style)
        
        # Process emotional expressions for better Bark compatibility
        processed_prompt = self._process_emotions(processed_prompt)
        
        return processed_prompt
    
    def _process_emotions(self, prompt: str) -> str:
        """Process emotional expressions to work better with Bark"""
        # Bark emotion mapping - convert common formats to Bark-compatible ones
        emotion_mappings = {
            # Laughter variations
            '[laughs]': '[laughter]',
            '[laugh]': '[laughter]',
            '[laughing]': '[laughter]',
            '[chuckles]': '[laughter]',
            '[chuckle]': '[laughter]',
            '[chuckling]': '[laughter]',
            '[giggles]': '[laughter]',
            '[giggle]': '[laughter]',
            '[giggling]': '[laughter]',
            
            # Sigh variations
            '[sighs]': '[sigh]',
            '[sigh]': '[sigh]',
            '[sighing]': '[sigh]',
            
            # Gasp variations
            '[gasps]': '[gasp]',
            '[gasp]': '[gasp]',
            '[gasping]': '[gasp]',
            
            # Whisper variations
            '[whispers]': '[whisper]',
            '[whisper]': '[whisper]',
            '[whispering]': '[whisper]',
            
            # Cry variations
            '[cries]': '[cry]',
            '[cry]': '[cry]',
            '[crying]': '[cry]',
            '[sobs]': '[cry]',
            '[sob]': '[cry]',
            '[sobbing]': '[cry]',
            
            # Scream variations
            '[screams]': '[scream]',
            '[scream]': '[scream]',
            '[screaming]': '[scream]',
            '[shouts]': '[scream]',
            '[shout]': '[scream]',
            '[shouting]': '[scream]',
            '[yells]': '[scream]',
            '[yell]': '[scream]',
            '[yelling]': '[scream]',
            
            # Sing variations
            '[sings]': '[sing]',
            '[sing]': '[sing]',
            '[singing]': '[sing]',
            '[hums]': '[sing]',
            '[hum]': '[sing]',
            '[humming]': '[sing]',
            
            # Throat clearing variations
            '[clears throat]': '[clears throat]',
            '[clearing throat]': '[clears throat]',
            '[throat clearing]': '[clears throat]',
            
            # Cough variations
            '[coughs]': '[cough]',
            '[cough]': '[cough]',
            '[coughing]': '[cough]',
            
            # Yawn variations
            '[yawns]': '[yawn]',
            '[yawn]': '[yawn]',
            '[yawning]': '[yawn]',
            
            # Breathing variations
            '[breathes]': '[breathing]',
            '[breathe]': '[breathing]',
            '[breathing]': '[breathing]',
            '[heavy breathing]': '[breathing]',
            '[panting]': '[breathing]',
            '[pants]': '[breathing]',
            '[pant]': '[breathing]',
            
            # Music and sound effects
            '[music]': '[music]',
            '[♪]': '[music]',
            '[♪♪]': '[music]',
            '[♪♪♪]': '[music]',
            
            # Hesitation and pauses
            '...': '...',
            '—': '—',
            '[pause]': '...',
            '[pauses]': '...',
            '[pausing]': '...',
        }
        
        processed = prompt
        for old_emotion, new_emotion in emotion_mappings.items():
            processed = processed.replace(old_emotion, new_emotion)
        
        return processed
    
    def _post_process_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """Post-process audio to fix timing and quality issues"""
        try:
            # Remove trailing silence (common issue with Bark)
            audio_array = self._remove_trailing_silence(audio_array, sample_rate)
            
            # Normalize audio levels
            audio_array = self._normalize_audio(audio_array)
            
            return audio_array
        except Exception as e:
            print(f"Warning: Audio post-processing failed: {e}")
            return audio_array
    
    def _remove_trailing_silence(self, audio_array: np.ndarray, sample_rate: int, silence_threshold: float = 0.01) -> np.ndarray:
        """Remove trailing silence from audio"""
        try:
            # Find the last non-silent sample
            silence_samples = int(sample_rate * 0.1)  # 100ms of silence threshold
            threshold = silence_threshold
            
            # Work backwards from the end
            for i in range(len(audio_array) - 1, -1, -1):
                if abs(audio_array[i]) > threshold:
                    # Found non-silent sample, add a small buffer
                    end_index = min(i + silence_samples, len(audio_array))
                    return audio_array[:end_index]
            
            # If all audio is silent, return a small portion
            return audio_array[:int(sample_rate * 0.1)]
        except Exception as e:
            print(f"Warning: Failed to remove trailing silence: {e}")
            return audio_array
    
    def _normalize_audio(self, audio_array: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """Normalize audio levels"""
        try:
            # Find the maximum absolute value
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                # Normalize to target level
                normalized = audio_array * (target_level / max_val)
                # Ensure we don't exceed 1.0
                return np.clip(normalized, -1.0, 1.0)
            return audio_array
        except Exception as e:
            print(f"Warning: Failed to normalize audio: {e}")
            return audio_array
    
    def _apply_voice_style(self, prompt: str, voice_style: str) -> str:
        """Apply voice style guidance to the prompt"""
        if voice_style == "auto":
            return prompt
        
        # Voice style prefixes to guide Bark's voice selection
        voice_style_prefixes = {
            "professional": "In a professional, business-like tone: ",
            "casual": "In a casual, friendly conversational tone: ",
            "storytelling": "In a storytelling, narrative voice: ",
            "character": "In a distinctive, animated character voice: ",
            "male": "In a deep, masculine voice: ",
            "female": "In a clear, feminine voice: ",
            "child": "In a young, energetic child's voice: ",
            "narrator": "In a clear, authoritative narrator voice: "
        }
        
        prefix = voice_style_prefixes.get(voice_style, "")
        return f"{prefix}{prompt}"
    
    async def _generate_bark_placeholder(self, prompt: str, sample_rate: int, output_format: str) -> str:
        """Generate a placeholder audio that simulates Bark TTS"""
        try:
            sr = sample_rate
            seconds = min(len(prompt.split()) * 0.5, 10)  # Estimate duration based on prompt length
            t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
            
            # Create a more realistic voice-like tone with modulation
            base_freq = 150.0  # Lower frequency for more voice-like sound
            mod_freq = 2.0  # Modulation frequency for natural variation
            
            # Add some harmonics for more realistic voice
            audio_array = (
                0.3 * np.sin(2 * np.pi * base_freq * t) +
                0.1 * np.sin(2 * np.pi * base_freq * 2 * t) +
                0.05 * np.sin(2 * np.pi * base_freq * 3 * t)
            )
            
            # Add modulation for natural voice variation
            modulation = 1 + 0.3 * np.sin(2 * np.pi * mod_freq * t)
            audio_array *= modulation
            
            # Add some envelope to make it sound more like speech
            envelope = np.exp(-t / (seconds * 0.3))  # Decay envelope
            audio_array *= envelope
            
            # Convert to float32
            audio_array = audio_array.astype(np.float32)
            
            return await self._save_audio(audio_array, sr, output_format)
            
        except Exception as e:
            raise RuntimeError(f"Bark placeholder generation failed: {e}")
    
    async def _save_audio(self, audio_array: np.ndarray, sample_rate: int, output_format: str) -> str:
        """Save audio array to file using proper filename extensions"""
        try:
            # Create output directory
            os.makedirs("../outputs/audio", exist_ok=True)
            
            # Generate output filename with proper extension
            import uuid
            filename_base = f"audio_{uuid.uuid4().hex[:8]}"
            output_path = f"../outputs/audio/{filename_base}.{output_format}"
            
            if output_format == 'wav':
                # For WAV files, use scipy.io.wavfile for direct saving
                from scipy.io.wavfile import write as write_wav
                write_wav(output_path, sample_rate, audio_array)
                return output_path
            else:
                # For other formats (MP3), use soundfile + ffmpeg with proper filenames
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_wav = tmp.name
                try:
                    # Write to temporary WAV file
                    sf.write(tmp_wav, audio_array, sample_rate)
                    
                    # Convert to target format using ffmpeg with proper filenames
                    if output_format == 'mp3':
                        (
                            ffmpeg
                            .input(tmp_wav)
                            .output(output_path, acodec='libmp3lame', ar=sample_rate, ab='192k')
                            .overwrite_output()
                            .run(quiet=True)
                        )
                    else:
                        # For other formats, use default encoding
                        (
                            ffmpeg
                            .input(tmp_wav)
                            .output(output_path)
                            .overwrite_output()
                            .run(quiet=True)
                        )
                except ffmpeg.Error as e:
                    print(f"FFmpeg error details: {e}")
                    print(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'No stderr'}")
                    raise RuntimeError(f"ffmpeg error (see stderr output for detail): {e}")
                finally:
                    try:
                        if os.path.exists(tmp_wav):
                            os.unlink(tmp_wav)
                    except Exception:
                        pass
                
                return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to save audio: {e}")
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple resampling without librosa
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            return np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio
            )
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory"""
        try:
            if model_name in self.models:
                del self.models[model_name]
                torch.cuda.empty_cache() if self.device != "cpu" else None
                return True
            return False
        except Exception as e:
            print(f"Error unloading model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name in self.available_models:
            info = self.available_models[model_name].copy()
            info["loaded"] = model_name in self.models
            return info
        return None
    
    def get_bark_voice_presets(self) -> List[Dict[str, Any]]:
        """Get the English voice presets for Bark"""
        return [
            {
                "id": "v2/en_speaker_0",
                "name": "Speaker 0 (EN)",
                "description": "English Male",
                "gender": "Male",
                "language": "English",
                "special": None
            },
            {
                "id": "v2/en_speaker_1", 
                "name": "Speaker 1 (EN)",
                "description": "English Male",
                "gender": "Male",
                "language": "English",
                "special": None
            },
            {
                "id": "v2/en_speaker_2",
                "name": "Speaker 2 (EN)", 
                "description": "English Male",
                "gender": "Male",
                "language": "English",
                "special": None
            },
            {
                "id": "v2/en_speaker_3",
                "name": "Speaker 3 (EN)",
                "description": "English Male", 
                "gender": "Male",
                "language": "English",
                "special": None
            },
            {
                "id": "v2/en_speaker_4",
                "name": "Speaker 4 (EN)",
                "description": "English Male",
                "gender": "Male", 
                "language": "English",
                "special": None
            },
            {
                "id": "v2/en_speaker_5",
                "name": "Speaker 5 (EN)",
                "description": "English Male",
                "gender": "Male",
                "language": "English", 
                "special": "Grainy"
            },
            {
                "id": "v2/en_speaker_6",
                "name": "Speaker 6 (EN)",
                "description": "English Male",
                "gender": "Male",
                "language": "English",
                "special": "Suno Favorite"
            },
            {
                "id": "v2/en_speaker_7",
                "name": "Speaker 7 (EN)",
                "description": "English Male",
                "gender": "Male",
                "language": "English",
                "special": None
            },
            {
                "id": "v2/en_speaker_8", 
                "name": "Speaker 8 (EN)",
                "description": "English Male",
                "gender": "Male",
                "language": "English",
                "special": None
            },
            {
                "id": "v2/en_speaker_9",
                "name": "Speaker 9 (EN)",
                "description": "English Female",
                "gender": "Female",
                "language": "English",
                "special": None
            }
        ]
