from gtts import gTTS

text = "asdf"
tts = gTTS(text=text, lang='en')
tts.save("output.mp3")