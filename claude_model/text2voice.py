from gtts import gTTS

text = "mother Fucker"
tts = gTTS(text=text, lang='en')
tts.save("output.mp3")