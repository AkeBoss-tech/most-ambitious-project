from youtube_transcript_api import YouTubeTranscriptApi

dictionary = YouTubeTranscriptApi.get_transcript("GfNbEZQUvIw")

with open("lyrics.txt", "w") as text_file:
    print(dictionary, file=text_file)