import os
from pathlib import Path
from instabot import Bot
from dotenv import load_dotenv
import time

load_dotenv()

class InstagramPoster:
    def __init__(self):
        self.username = os.getenv("INSTAGRAM_USERNAME")
        self.password = os.getenv("INSTAGRAM_PASSWORD")
        self.bot = None
        self.assets_dir = Path("assets")
        
    def login(self):
        self.bot = Bot()
        self.bot.login(username=self.username, password=self.password)
        
    def post_video(self, video_path, caption=""):
        if not self.bot:
            self.login()
            
        try:
            self.bot.upload_video(str(video_path), caption=caption)
            return True
        except Exception as e:
            print(f"Error posting video: {e}")
            return False
            
    def monitor_and_post(self):
        while True:
            # Look for new videos in assets directory
            videos = list(self.assets_dir.glob("final_*.mp4"))
            
            for video in videos:
                if self.post_video(video):
                    # Move posted video to archive or delete
                    os.remove(video)
                    
            # Wait before next check
            time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    poster = InstagramPoster()
    poster.monitor_and_post()
