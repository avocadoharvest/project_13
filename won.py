# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # 1. PDF 로드
# loader = PyPDFLoader("C:\_vscode\Chap.03\knudata\project_13\토스 파트 3 문제.pdf")
# pages = loader.load()

# # 2. 텍스트 청크 나누기 (문제 단위로 분리되게 기준 조정)
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50,
#     separators=["\n\n", "\n", ".", "?"]
# )
# docs = splitter.split_documents(pages)

# # 3. 출력 확인
# for i, doc in enumerate(docs[:5]):
#     print(f"[문제 {i+1}]\n", doc.page_content)

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import whisper
# import speech_recognition as sr
# import numpy as np

# # ===== 1. PDF에서 문제 추출 =====
# pdf_path = r"C:\_vscode\Chap.03\knudata\project_13\토스 파트 3 문제.pdf"
# loader = PyPDFLoader(pdf_path)
# pages = loader.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50,
#     separators=["\n\n", "\n", ".", "?"]
# )
# docs = splitter.split_documents(pages)

# questions = []
# for doc in docs:
#     lines = doc.page_content.strip().split("\n")
#     for line in lines:
#         if line.startswith("Q") and "?" in line:
#             questions.append(line.strip())

# # ===== 2. Whisper + Mic 준비 =====
# model = whisper.load_model("base")
# r = sr.Recognizer()

# # ===== 3. 반복: 문제 보여주고 마이크로 받기 =====
# for i, question in enumerate(questions, 1):
#     print(f"\n🎯 [문제 {i}] {question}")
#     print("🎤 답변을 말해주세요...")

#     with sr.Microphone(sample_rate=16000) as source:
#         audio = r.listen(source)
#         data = audio.get_wav_data()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import whisper
import speech_recognition as sr
import numpy as np

def run_toeic_part3_stt(pdf_path: str):
    # ===== 1. PDF에서 문제 추출 =====
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "?"]
    )
    docs = splitter.split_documents(pages)

    questions = []
    for doc in docs:
        lines = doc.page_content.strip().split("\n")
        for line in lines:
            if line.startswith("Q") and "?" in line:
                questions.append(line.strip())

    # ===== 2. Whisper + Mic 준비 =====
    model = whisper.load_model("base")
    r = sr.Recognizer()

    # ===== 3. 반복: 문제 보여주고 마이크로 받고 결과 저장 =====
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n🎯 [문제 {i}] {question}")
        print("🎤 답변을 말해주세요...")

        with sr.Microphone(sample_rate=16000) as source:
            audio = r.listen(source)
            data = audio.get_wav_data()
            wav = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0

            print("🧠 음성 인식 중...")
            result = model.transcribe(wav, fp16=False)
            print("📝 인식된 답변:", result["text"])
            results.append((question, result["text"]))

        input("👉 [엔터]를 누르면 다음 문제로 넘어갑니다.")

    return results


if __name__ == "__main__":
    pdf_path = r"C:\_vscode\Chap.03\knudata\project_13\토스 파트 3 문제.pdf"
    results = run_toeic_part3_stt(pdf_path)

    print("\n🎯 모든 답변 결과 요약:")
    for i, (q, a) in enumerate(results, 1):
        print(f"\n[문제 {i}] {q}")
        print(f"🗣️ 답변: {a}")
