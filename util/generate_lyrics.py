import speech_recognition as sr

r = sr.Recognizer()

song = sr.AudioFile('./files/audio/Havana.wav')

with song as source:
    # r.adjust_for_ambient_noise(source)
    audio = r.record(source)

print(type(audio))

lyrics = r.recognize_sphinx(audio)
with open("output.txt", "w") as text_file:
    print(lyrics, file=text_file)