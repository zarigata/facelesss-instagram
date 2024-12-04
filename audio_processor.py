import os
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import random
import numpy as np

class AudioProcessor:
    def __init__(self):
        self.assets_dir = Path("assets")
        self.music_dir = self.assets_dir / "music"
        self.music_dir.mkdir(exist_ok=True)
        
    def get_random_music(self):
        """Get a random background music track"""
        music_extensions = {'.mp3', '.wav', '.m4a'}
        music_files = [
            f for f in self.music_dir.glob('*')
            if f.suffix.lower() in music_extensions
        ]
        
        if not music_files:
            raise Exception("No background music found in assets/music directory")
            
        return random.choice(music_files)
    
    def normalize_audio(self, audio_clip, target_db=-30):
        """Normalize audio to target dB level"""
        # Get the current dB level (approximated through RMS)
        samples = audio_clip.to_soundarray()
        rms = np.sqrt(np.mean(samples**2))
        current_db = 20 * np.log10(rms)
        
        # Calculate the scaling factor needed to reach target dB
        db_change = target_db - current_db
        scaling_factor = 10 ** (db_change / 20)
        
        return audio_clip.volumex(scaling_factor)

    def add_background_music(self, video_path, volume=0.3):
        """Add background music to video at specified volume"""
        try:
            # Load video
            video = VideoFileClip(str(video_path))
            
            # Get random music track
            music_path = self.get_random_music()
            music = AudioFileClip(str(music_path))
            
            # Loop music if needed
            if music.duration < video.duration:
                repeats = int(video.duration / music.duration) + 1
                music = music.loop(repeats)
            
            # Trim music to video length and normalize to -30dB
            music = music.set_duration(video.duration)
            music = self.normalize_audio(music, target_db=-30)
            
            # Mix original audio with background music
            final_audio = CompositeVideoClip([
                video.set_audio(video.audio.volumex(1.0)),
                video.set_audio(music)
            ]).audio
            
            # Create output path
            output_path = video_path.parent / f"{video_path.stem}_with_music{video_path.suffix}"
            
            # Create final video with mixed audio
            final_video = video.set_audio(final_audio)
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac'
            )
            
            # Clean up
            video.close()
            music.close()
            
            return output_path
            
        except Exception as e:
            print(f"Error adding background music: {str(e)}")
            raise
