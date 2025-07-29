import gradio as gr
import os
import tempfile
from dotenv import load_dotenv
from gtts import gTTS

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from text2speech import speech_to_text   # 파일 경로에서 텍스트로 변환 함수 필요

# 환경변수 및 OpenAI API 키 로딩
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

questions = []
pdf_fulltext = ""
question_index = 0
evaluate_chain = None

def extract_questions(pdf_file):
    global questions, question_index, pdf_fulltext
    if pdf_file is None:
        return "❌ PDF 파일을 업로드하세요."
    pdf_path = pdf_file.name
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", "?"])
    docs = splitter.split_documents(pages)
    extracted = []
    for doc in docs:
        lines = doc.page_content.strip().split("\n")
        for line in lines:
            if line.strip().startswith("Q") and "?" in line:
                extracted.append(line.strip())
    questions = extracted
    question_index = 0
    pdf_fulltext = "\n".join([doc.page_content for doc in pages])
    return f"✅ 총 {len(questions)}개의 문제가 추출되었습니다."

def setup_chain():
    global evaluate_chain
    prompt = PromptTemplate.from_template("""
#Question: {question}
#Answer: {answer}
#Context: {context}

[출력 예시]   
점수: B점   
피드백: 답변이 핵심 내용을 잘 언급했으나, 세부정보가 빠짐.
""")
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        streaming=False,
    )
    evaluate_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )
    return "✅ LLM 채점 체인 준비 완료!"

def get_next_question():
    global question_index
    if not questions:
        return "❗ 먼저 PDF를 업로드해주세요.", None
    if question_index >= len(questions):
        return "🎉 모든 문제를 완료했습니다!", None

    question = questions[question_index]
    question_index += 1

    tts = gTTS(text=question, lang='en')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    tmp.close()
    return f"[문제 {question_index}] {question}", tmp.name

def evaluate_audio_response(audio):
    global question_index, evaluate_chain, pdf_fulltext
    if evaluate_chain is None:
        return "❌ 평가 체인을 먼저 준비해주세요."
    if question_index == 0:
        return "❗ 문제를 먼저 시작해주세요."

    question = questions[question_index - 1]
    user_answer = speech_to_text(audio)   # 오디오 파일로부터 텍스트 추출

    # 전체 pdf context 사용 (없으면 context="")
    context = pdf_fulltext if pdf_fulltext else ""

    result = evaluate_chain.invoke({
        "question": question,
        "answer": user_answer,
        "context": context,
    })
    return f"🧠 AI 평가 결과:\n\n{result['text'] if isinstance(result, dict) and 'text' in result else result}"

with gr.Blocks() as demo:
    gr.Markdown("## 🗣️ 토익스피킹 AI 채점 시스템 (LLM Only 버전)")

    with gr.Row():
        pdf_input = gr.File(label="📄 문제 PDF 업로드", file_types=[".pdf"])
        upload_result = gr.Textbox(label="문제 추출 결과")
    with gr.Row():
        setup_btn = gr.Button("🧠 평가 체인 준비")
        chain_status = gr.Textbox(label="평가 체인 상태")
    next_q_btn = gr.Button("📢 다음 문제")
    question_output = gr.Textbox(label="문제 보기")
    question_audio = gr.Audio(label="문제 음성", type="filepath")
    audio_input = gr.Audio(label="🎤 음성 답변", type="filepath")
    result_output = gr.Textbox(label="📊 AI 평가 결과")

    pdf_input.change(fn=extract_questions, inputs=[pdf_input], outputs=[upload_result])
    setup_btn.click(fn=setup_chain, outputs=[chain_status])
    next_q_btn.click(fn=get_next_question, outputs=[question_output, question_audio])
    audio_input.change(fn=evaluate_audio_response, inputs=[audio_input], outputs=[result_output])

demo.launch()
