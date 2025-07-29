import whisper

def speech_to_text(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language='en', fp16=False)
    return result['text']
