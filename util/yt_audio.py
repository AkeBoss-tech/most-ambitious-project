# importing packages
from pytube import YouTube
import os

def download_video(url, path="./files/audio"):
    # url input from user
    yt = YouTube(url)
    
    # extract only audio
    video = yt.streams.filter(only_audio=True).first()
    
    # download the file
    out_file = video.download(output_path=path)
    
    # save the file
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    
    # result of success
    print(yt.title + " has been successfully downloaded.")

if __name__ == "__main__":
    # Havana
    download_video("https://www.youtube.com/watch?v=GfNbEZQUvIw")