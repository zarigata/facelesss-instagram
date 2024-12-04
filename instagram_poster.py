import os
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from instabot import Bot
from video_generator import VideoGenerator
import glob
import json
import random

# Set up logging
logging.basicConfig(
    filename='instagram_poster.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class InstagramPoster:
    def __init__(self):
        load_dotenv()
        self.username = os.getenv('INSTAGRAM_USERNAME')
        self.password = os.getenv('INSTAGRAM_PASSWORD')
        self.assets_dir = Path("assets")
        self.bot = None
        
        # Load config
        with open("config.json", "r") as f:
            self.config = json.load(f)
    
    def generate_caption(self, topic, content):
        """Generate caption using Ollama"""
        try:
            ollama_url = f"http://{self.config['ollama']['host']}:{self.config['ollama']['port']}/api/generate"
            
            prompt = f"""Generate an engaging Instagram caption for a video about {topic}.
            The video content is about: {content}
            
            Requirements:
            1. Start with an engaging hook
            2. Include 5-10 relevant hashtags
            3. Use emojis naturally
            4. Keep it under 200 characters (not counting hashtags)
            5. Make hashtags specific to men's self-improvement and alpha male content
            6. Add these hashtags at the end: #alphamale #masculine #sigma
            
            Format the response as a caption ready to post."""
            
            response = requests.post(ollama_url, json={
                "model": self.config['ollama']['model'],
                "prompt": prompt,
                "stream": False
            })
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.text}")
            
            result = response.json()
            caption = result.get('response', '').strip()
            
            # Fallback caption if generation fails
            if not caption:
                caption = "Level up your life 💪 #alphamale #masculine #sigma #mindset #growth"
            
            logging.info(f"Generated caption: {caption}")
            return caption
            
        except Exception as e:
            logging.error(f"Error generating caption: {str(e)}")
            return "Level up your life 💪 #alphamale #masculine #sigma #mindset #growth"
    
    def cleanup_files(self, final_video_path):
        """Clean up temporary files, keeping only the final video"""
        try:
            # Remove all temporary mp3 files
            for audio_file in self.assets_dir.glob("*.mp3"):
                if audio_file.exists():
                    audio_file.unlink()
            
            # Remove all temporary mp4 files except the final video
            for video_file in self.assets_dir.glob("*.mp4"):
                if video_file.exists() and video_file != final_video_path:
                    video_file.unlink()
            
            # Remove cookie files from instabot
            cookie_files = glob.glob("config/*cookie.json")
            for cookie_file in cookie_files:
                if os.path.exists(cookie_file):
                    os.remove(cookie_file)
                    
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
    
    def login(self):
        """Login to Instagram with retry logic"""
        try:
            if self.bot is not None:
                self.bot.logout()
            
            self.bot = Bot()
            self.bot.login(username=self.username, password=self.password)
            return True
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            return False
    
    def post_video(self, video_path, topic, content, max_retries=3):
        """Post video to Instagram with retry logic"""
        caption = self.generate_caption(topic, content)
        
        for attempt in range(max_retries):
            try:
                if not self.login():
                    raise Exception("Failed to login")
                
                success = self.bot.upload_video(
                    video_path,
                    caption=caption,
                    retry_count=1
                )
                
                if success:
                    logging.info(f"Successfully posted video: {video_path}")
                    return True
                else:
                    raise Exception("Upload returned False")
                    
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(60)  # Wait 1 minute before retrying
                    continue
                return False
    
    def run(self):
        """Main execution loop"""
        while True:
            try:
                # Generate video
                generator = VideoGenerator()
                topic = random.choice(self.config["topics"])
                content = generator.generate_content(topic)
                video_path = generator.generate_video()
                
                if not video_path or not Path(video_path).exists():
                    raise Exception("Video generation failed")
                
                # Clean up temporary files
                self.cleanup_files(Path(video_path))
                
                # Try to post the video
                if self.post_video(str(video_path), topic, content):
                    logging.info("Video posted successfully")
                else:
                    logging.error("Failed to post video after all retries")
                
                # Clean up the final video after posting
                if Path(video_path).exists():
                    Path(video_path).unlink()
                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
            
            # Wait for the configured interval before next run
            interval_hours = float(self.config["video"]["generation_interval"].replace("h", ""))
            logging.info(f"Waiting {interval_hours} hours before next run")
            time.sleep(interval_hours * 3600)

if __name__ == "__main__":
    poster = InstagramPoster()
    poster.run()
