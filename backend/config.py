"""
Configuration management for the text-to-media application.
Handles paths, environment variables, and cross-platform compatibility.
"""

import os
import pathlib
from typing import Optional


class Config:
    """Application configuration with environment-aware path resolution."""
    
    def __init__(self):
        # Base directory - this will be the project root
        self.base_dir = pathlib.Path(__file__).parent.parent.resolve()
        
        # Get configuration from environment variables or use defaults
        self.outputs_dir = self._get_path_from_env('OUTPUTS_DIR', 'outputs')
        self.models_dir = self._get_path_from_env('MODELS_DIR', 'models')
        self.temp_dir = self._get_path_from_env('TEMP_DIR', 'temp')
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _get_path_from_env(self, env_var: str, default: str) -> pathlib.Path:
        """Get path from environment variable or use default relative to base directory."""
        env_path = os.getenv(env_var)
        if env_path:
            # If absolute path provided, use it directly
            if pathlib.Path(env_path).is_absolute():
                return pathlib.Path(env_path)
            # If relative path provided, make it relative to base directory
            else:
                return self.base_dir / env_path
        else:
            # Use default relative to base directory
            return self.base_dir / default
    
    def _ensure_directories(self):
        """Create directories if they don't exist."""
        for dir_path in [self.outputs_dir, self.models_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def outputs_path(self) -> pathlib.Path:
        """Get the outputs directory path."""
        return self.outputs_dir
    
    @property
    def models_path(self) -> pathlib.Path:
        """Get the models directory path."""
        return self.models_dir
    
    @property
    def temp_path(self) -> pathlib.Path:
        """Get the temporary directory path."""
        return self.temp_dir
    
    @property
    def videos_output_path(self) -> pathlib.Path:
        """Get the videos output directory path."""
        videos_dir = self.outputs_dir / 'videos'
        videos_dir.mkdir(parents=True, exist_ok=True)
        return videos_dir
    
    @property
    def audio_output_path(self) -> pathlib.Path:
        """Get the audio output directory path."""
        audio_dir = self.outputs_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        return audio_dir
    
    @property
    def image_output_path(self) -> pathlib.Path:
        """Get the image output directory path (same as outputs_dir)."""
        return self.outputs_dir
    
    @property
    def voice_previews_path(self) -> pathlib.Path:
        """Get the voice previews directory path."""
        previews_dir = self.outputs_dir / 'voice-previews'
        previews_dir.mkdir(parents=True, exist_ok=True)
        return previews_dir
    
    @property
    def custom_voices_path(self) -> pathlib.Path:
        """Get the custom voices directory path."""
        voices_dir = self.outputs_dir / 'custom-voices'
        voices_dir.mkdir(parents=True, exist_ok=True)
        return voices_dir
    
    def get_model_path(self, model_type: str, model_name: str) -> pathlib.Path:
        """Get the path for a specific model."""
        return self.models_dir / model_type / model_name
    
    def get_relative_path(self, absolute_path: pathlib.Path) -> str:
        """Convert absolute path to relative path for API responses."""
        try:
            return str(absolute_path.relative_to(self.base_dir))
        except ValueError:
            # If path is not relative to base_dir, return the absolute path as string
            return str(absolute_path)
    
    def get_absolute_path(self, relative_path: str) -> pathlib.Path:
        """Convert relative path to absolute path."""
        if pathlib.Path(relative_path).is_absolute():
            return pathlib.Path(relative_path)
        return self.base_dir / relative_path
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for API responses."""
        return {
            "base_dir": str(self.base_dir),
            "outputs_dir": str(self.outputs_dir),
            "models_dir": str(self.models_dir),
            "temp_dir": str(self.temp_dir),
            "outputs_path": str(self.outputs_path),
            "models_path": str(self.models_path),
            "videos_output_path": str(self.videos_output_path),
            "audio_output_path": str(self.audio_output_path),
            "image_output_path": str(self.image_output_path),
            "voice_previews_path": str(self.voice_previews_path),
            "custom_voices_path": str(self.custom_voices_path),
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def update_config_from_env():
    """Update configuration from environment variables."""
    global config
    config = Config()


# Environment variable documentation
ENV_VARS_DOC = """
Environment Variables for Configuration:

OUTPUTS_DIR: Directory for generated outputs (videos, audio, images)
             Default: 'outputs' (relative to project root)
             Example: OUTPUTS_DIR=/path/to/outputs
             Example: OUTPUTS_DIR=./my-outputs

MODELS_DIR: Directory for AI models
            Default: 'models' (relative to project root)
            Example: MODELS_DIR=/path/to/models
            Example: MODELS_DIR=./my-models

TEMP_DIR: Directory for temporary files
          Default: 'temp' (relative to project root)
          Example: TEMP_DIR=/tmp/my-app
          Example: TEMP_DIR=./temp

Note: All paths can be either absolute or relative to the project root.
"""
