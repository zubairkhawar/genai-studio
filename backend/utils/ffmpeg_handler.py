import ffmpeg
import os
from typing import Optional

class FFmpegHandler:
    """Handle video and audio conversion using FFmpeg"""
    
    def __init__(self):
        self.check_ffmpeg()
    
    def check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            ffmpeg.probe('dummy')
        except ffmpeg.Error:
            # This is expected for a dummy file, but it means ffmpeg is available
            pass
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    def convert_video(self, input_path: str, output_path: str, 
                     format: str = "mp4", quality: str = "high") -> str:
        """Convert video to specified format"""
        try:
            # Set quality based on format
            if quality == "high":
                if format == "mp4":
                    vcodec = "libx264"
                    crf = "18"
                elif format == "webm":
                    vcodec = "libvpx-vp9"
                    crf = "18"
                else:
                    vcodec = "libx264"
                    crf = "18"
            else:  # medium quality
                if format == "mp4":
                    vcodec = "libx264"
                    crf = "23"
                elif format == "webm":
                    vcodec = "libvpx-vp9"
                    crf = "23"
                else:
                    vcodec = "libx264"
                    crf = "23"
            
            # Convert video
            (
                ffmpeg
                .input(input_path)
                .output(output_path, vcodec=vcodec, crf=crf, acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
            
            return output_path
            
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e}")
    
    def convert_audio(self, input_path: str, output_path: str, 
                     format: str = "wav", sample_rate: int = 22050) -> str:
        """Convert audio to specified format"""
        try:
            if format == "wav":
                acodec = "pcm_s16le"
            elif format == "mp3":
                acodec = "libmp3lame"
                bitrate = "192k"
            elif format == "flac":
                acodec = "flac"
            else:
                acodec = "pcm_s16le"
            
            # Build conversion command
            output_args = {
                'acodec': acodec,
                'ar': sample_rate
            }
            
            if format == "mp3":
                output_args['ab'] = bitrate
            
            # Convert audio
            (
                ffmpeg
                .input(input_path)
                .output(output_path, **output_args)
                .overwrite_output()
                .run(quiet=True)
            )
            
            return output_path
            
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e}")
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video information"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'video'), None)
            
            if video_stream:
                return {
                    'duration': float(probe['format']['duration']),
                    'width': int(video_stream['width']),
                    'height': int(video_stream['height']),
                    'fps': eval(video_stream['r_frame_rate']),
                    'codec': video_stream['codec_name']
                }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}
    
    def get_audio_info(self, audio_path: str) -> dict:
        """Get audio information"""
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'audio'), None)
            
            if audio_stream:
                return {
                    'duration': float(probe['format']['duration']),
                    'sample_rate': int(audio_stream['sample_rate']),
                    'channels': int(audio_stream['channels']),
                    'codec': audio_stream['codec_name']
                }
        except Exception as e:
            print(f"Error getting audio info: {e}")
            return {}
    
    def create_thumbnail(self, video_path: str, thumbnail_path: str, 
                        time_offset: float = 1.0) -> str:
        """Create thumbnail from video"""
        try:
            (
                ffmpeg
                .input(video_path, ss=time_offset)
                .output(thumbnail_path, vframes=1, format='image2')
                .overwrite_output()
                .run(quiet=True)
            )
            return thumbnail_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Thumbnail creation failed: {e}")
    
    def trim_video(self, input_path: str, output_path: str, 
                   start_time: float, duration: float) -> str:
        """Trim video to specified duration"""
        try:
            (
                ffmpeg
                .input(input_path, ss=start_time, t=duration)
                .output(output_path, vcodec='copy', acodec='copy')
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Video trimming failed: {e}")
    
    def trim_audio(self, input_path: str, output_path: str, 
                   start_time: float, duration: float) -> str:
        """Trim audio to specified duration"""
        try:
            (
                ffmpeg
                .input(input_path, ss=start_time, t=duration)
                .output(output_path, acodec='copy')
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Audio trimming failed: {e}")
