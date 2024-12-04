import os
import json
import time
import random
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips, concatenate_audioclips, vfx
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gtts import gTTS
import subprocess
import textwrap
from PIL.Image import Resampling
import re

# Load environment variables
load_dotenv()

class VideoGenerator:
    def __init__(self):
        self.config = self.load_config()
        self.assets_dir = Path("assets")
        self.assets_dir.mkdir(exist_ok=True)
        self.backgrounds_dir = self.assets_dir / "backgrounds"
        self.backgrounds_dir.mkdir(exist_ok=True)
        
    def load_config(self):
        with open("config.json", "r") as f:
            return json.load(f)
    
    def split_into_phrases(self, text, min_words=3, max_words=6):
        """Split text into phrases of 3-6 words, trying to break at natural points"""
        # First, split the text into sentences
        sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
        
        phrases = []
        for sentence in sentences:
            # Split sentence into words
            words = sentence.split()
            
            # Process each chunk of words
            current_chunk = []
            for word in words:
                current_chunk.append(word)
                
                # Check if we've reached max words or hit a natural break point
                chunk_size = len(current_chunk)
                if chunk_size >= min_words and (
                    chunk_size >= max_words or  # Max words reached
                    word.endswith((',', ';', ':')) or  # Natural break point
                    chunk_size >= min_words and len(words) - len(current_chunk) <= 2  # Near end of sentence
                ):
                    phrases.append(' '.join(current_chunk))
                    current_chunk = []
            
            # Add any remaining words
            if current_chunk:
                if len(current_chunk) < min_words and phrases:
                    # Combine with previous phrase if too short
                    phrases[-1] = phrases[-1] + ' ' + ' '.join(current_chunk)
                else:
                    phrases.append(' '.join(current_chunk))
        
        return phrases
    
    def estimate_phrase_duration(self, phrase):
        """Estimate the duration needed to speak a phrase"""
        # Rough estimation: average person speaks 2-3 words per second
        words = len(phrase.split())
        base_duration = words / 2.5  # Assuming 2.5 words per second
        return max(1.5, base_duration)  # Minimum 1.5 seconds per phrase
    
    def resize_frame(self, frame, newsize):
        """Custom resize function using modern Pillow constants"""
        img = Image.fromarray(frame)
        resized = img.resize(newsize, Resampling.LANCZOS)
        return np.array(resized)
    
    def resize_video(self, clip, width=None, height=None):
        """Custom video resize function"""
        if width is None and height is None:
            raise ValueError("At least one of width or height must be specified")
            
        w, h = clip.size
        
        if width is not None and height is None:
            height = int(h * width / w)
        elif height is not None and width is None:
            width = int(w * height / h)
            
        newsize = (int(width), int(height))
        
        def resize_frame(frame):
            return self.resize_frame(frame, newsize)
            
        return clip.fl_image(resize_frame)
    
    def get_background_clips(self, total_duration):
        """Get multiple background videos that total the required duration"""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        background_videos = [
            f for f in self.backgrounds_dir.glob('*')
            if f.suffix.lower() in video_extensions
        ]
        
        if not background_videos:
            raise Exception("No background videos found in assets/backgrounds directory")
        
        clips = []
        current_duration = 0
        used_videos = set()
        
        while current_duration < total_duration:
            # Get available videos (excluding used ones)
            available_videos = [v for v in background_videos if v not in used_videos]
            
            # If we've used all videos, reset the used videos set
            if not available_videos:
                used_videos.clear()
                available_videos = background_videos
            
            # Select random video
            video_path = random.choice(available_videos)
            used_videos.add(video_path)
            
            clip = VideoFileClip(str(video_path))
            
            # Calculate how much duration we still need
            remaining_duration = total_duration - current_duration
            
            # If this clip is longer than needed, trim it
            if clip.duration > remaining_duration:
                clip = clip.subclip(0, remaining_duration)
            
            clips.append(clip)
            current_duration += clip.duration
        
        return clips
    
    def fit_text_to_box(self, text, font, max_width, max_height):
        """Fit text to a bounding box by adjusting font size"""
        font_size = 100  # Start with large font size
        wrapped_lines = []
        
        while font_size > 20:  # Minimum font size
            font = ImageFont.truetype(font.path, font_size)
            wrapped_lines = []
            
            # Calculate average characters per line based on width
            avg_char_width = font.getlength('x')
            chars_per_line = int(max_width / avg_char_width)
            
            # Wrap text
            lines = textwrap.wrap(text, width=chars_per_line)
            
            # Calculate total height
            bbox = font.getbbox('hg')
            line_height = (bbox[3] - bbox[1]) * 1.2  # Add 20% line spacing
            total_height = line_height * len(lines)
            
            if total_height <= max_height:
                wrapped_lines = lines
                break
                
            font_size -= 5
            
        return font, wrapped_lines, line_height
    
    def create_text_frame(self, text, size=(720, 1280)):
        """Create a frame with text that fits the video dimensions"""
        # Create a transparent background
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Try to use DejaVu Sans font, fall back to default if not available
        try:
            base_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            base_font = ImageFont.load_default()
            
        # Calculate maximum text box size (80% of video dimensions)
        max_width = int(size[0] * 0.8)
        max_height = int(size[1] * 0.8)
        
        # Fit text to box
        font, lines, line_height = self.fit_text_to_box(text, base_font, max_width, max_height)
        
        # Calculate total text height
        total_height = line_height * len(lines)
        
        # Calculate starting Y position to center text vertically
        current_y = (size[1] - total_height) // 2
        
        # Draw each line
        for line in lines:
            # Center each line horizontally
            line_width = font.getlength(line)
            x = (size[0] - line_width) // 2
            
            # Draw text with black outline
            outline_width = 2
            for offset_x, offset_y in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.text(
                    (x + offset_x * outline_width, current_y + offset_y * outline_width),
                    line,
                    font=font,
                    fill='black'
                )
            
            # Draw main text
            draw.text((x, current_y), line, font=font, fill='white')
            current_y += line_height
            
        return np.array(image)
    
    def get_random_outro_image(self):
        """Get a random image from the images directory"""
        images_dir = self.assets_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        images = [
            f for f in images_dir.glob('*')
            if f.suffix.lower() in image_extensions
        ]
        
        if not images:
            raise Exception("No images found in assets/images directory")
            
        return random.choice(images)
    
    def create_outro(self, duration=4):
        """Create an outro clip with a random image"""
        try:
            # Get random image
            image_path = self.get_random_outro_image()
            image = Image.open(str(image_path))
            
            # Resize image to match video dimensions (720x1280)
            target_size = (720, 1280)
            
            # Calculate resize dimensions maintaining aspect ratio
            aspect_ratio = image.width / image.height
            target_ratio = target_size[0] / target_size[1]
            
            if aspect_ratio > target_ratio:
                # Image is wider than target
                new_height = target_size[1]
                new_width = int(new_height * aspect_ratio)
            else:
                # Image is taller than target
                new_width = target_size[0]
                new_height = int(new_width / aspect_ratio)
            
            # Resize image
            image = image.resize((new_width, new_height), Resampling.LANCZOS)
            
            # Create new image with black background
            final_image = Image.new('RGB', target_size, 'black')
            
            # Paste resized image in center
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            final_image.paste(image, (paste_x, paste_y))
            
            # Convert to numpy array for MoviePy
            image_array = np.array(final_image)
            
            # Create clip with fade in
            outro_clip = (ImageClip(image_array)
                         .set_duration(duration)
                         .fadein(0.5)
                         .fadeout(0.5))
            
            return outro_clip
            
        except Exception as e:
            print(f"Error creating outro: {str(e)}")
            raise
    
    def generate_content(self, topic):
        # Connect to Ollama
        ollama_url = f"http://{self.config['ollama']['host']}:{self.config['ollama']['port']}/api/generate"
        prompt = self.config['video']['prompt_template'].format(topic=topic)
        
        response = requests.post(ollama_url, json={
            "model": self.config['ollama']['model'],
            "prompt": prompt,
            "stream": False
        })
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")
            
        try:
            result = response.json()
            if isinstance(result, dict) and 'response' in result:
                return result['response']
            elif isinstance(result, list) and result:
                return ' '.join(r.get('response', '') for r in result if isinstance(r, dict))
            else:
                raise Exception(f"Unexpected response format: {result}")
        except json.JSONDecodeError as e:
            print(f"Raw response: {response.text}")
            raise Exception(f"Failed to parse Ollama response: {e}")
    
    def generate_audio(self, text, language):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = self.assets_dir / f"audio_{timestamp}.mp3"
        
        tts = gTTS(text=text, lang=language)
        tts.save(str(audio_file))
        
        return audio_file
    
    def create_video_with_subtitles(self, text, audio_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.assets_dir / f"video_{timestamp}.mp4"
        
        try:
            # Split text into smaller phrases
            phrases = self.split_into_phrases(text)
            
            # Generate audio for each phrase and get durations
            audio_clips = []
            for phrase in phrases:
                audio = self.generate_audio(phrase, self.config["video"]["language"])
                audio_clip = AudioFileClip(str(audio))
                audio_clips.append((phrase, audio_clip))
            
            # Calculate total duration
            total_duration = sum(clip[1].duration for clip in audio_clips)
            
            # Get background video clips
            background_clips = self.get_background_clips(total_duration)
            
            # Process each background clip
            processed_bg_clips = []
            for bg_clip in background_clips:
                # Calculate new width while maintaining aspect ratio
                new_height = 1280
                aspect_ratio = bg_clip.w / bg_clip.h
                new_width = int(new_height * aspect_ratio)
                
                # Use custom resize function
                bg_clip = self.resize_video(bg_clip, width=new_width, height=new_height)
                
                # Center crop to 720x1280 (Instagram portrait)
                if new_width > 720:
                    crop_x = (new_width - 720) // 2
                    bg_clip = bg_clip.crop(x1=crop_x, y1=0, x2=crop_x+720, y2=1280)
                else:
                    # If video is too narrow, add black padding
                    pad_width = (720 - new_width) // 2
                    bg_clip = bg_clip.margin(left=pad_width, right=pad_width)
                
                processed_bg_clips.append(bg_clip)
            
            # Concatenate background clips
            background_clip = concatenate_videoclips(processed_bg_clips)
            
            # Create video clips for each phrase
            text_clips = []
            current_time = 0
            final_audio = None
            
            for phrase, audio_clip in audio_clips:
                frame = self.create_text_frame(phrase)
                duration = audio_clip.duration
                
                clip = (ImageClip(frame)
                       .set_duration(duration)
                       .set_position('center')
                       .set_start(current_time))
                
                text_clips.append(clip)
                
                # Concatenate audio clips
                if final_audio is None:
                    final_audio = audio_clip
                else:
                    final_audio = concatenate_audioclips([final_audio, audio_clip])
                
                current_time += duration
            
            # Create outro clip
            outro_clip = self.create_outro(4)  # 4 seconds outro
            
            # Composite main video
            main_clip = CompositeVideoClip(
                [background_clip] + text_clips,
                size=(720, 1280)
            ).set_audio(final_audio)
            
            # Concatenate main video with outro
            final_clip = concatenate_videoclips([
                main_clip,
                outro_clip
            ])
            
            # Write video file
            final_clip.write_videofile(
                str(output_file),
                fps=30,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Clean up temporary audio files and video clips
            for _, audio_clip in audio_clips:
                audio_clip.close()
            for clip in background_clips:
                clip.close()
            outro_clip.close()
            
            return output_file
            
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            raise
    
    def add_background_music(self, video_file):
        # For now, just return the original video file
        # TODO: Implement background music addition
        return video_file
    
    def generate_video(self):
        print("Starting video generation...")
        topic = random.choice(self.config["topics"])
        print(f"Selected topic: {topic}")
        
        content = self.generate_content(topic)
        print("Generated content from Ollama")
        
        video_file = self.create_video_with_subtitles(content, None)
        print(f"Created video with subtitles: {video_file}")
        
        # Add background music
        from audio_processor import AudioProcessor
        audio_processor = AudioProcessor()
        final_video = audio_processor.add_background_music(Path(video_file))
        print(f"Added background music: {final_video}")
        
        return final_video

if __name__ == "__main__":
    generator = VideoGenerator()
    final_video = generator.generate_video()
