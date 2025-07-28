from gtts import gTTS

text = "안녕하세요, 이 문장이 mp3 오디오로 변환됩니다."
tts = gTTS(text=text, lang='ko')
tts.save("output.mp3")