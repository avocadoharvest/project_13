import gradio as gr
import os
import tempfile
from dotenv import load_dotenv
from gtts import gTTS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from text2speech import speech_to_text  # 반드시 파일로부터(stt) 인식하는 구조로 되어 있어야 함

# 🚩 NEW: OpenAI 모듈 불러오기
import openai
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()

# 🚩 NEW: OpenAI API 키 환경변수에서 불러오기
openai.api_key = os.getenv("OPENAI_API_KEY")

vectorstore = None
questions = []
question_index = 0
evaluate_chain = None

def load_vectorstore():
    global vectorstore
    db_path = "./db/faiss"
    if not os.path.exists(db_path):
        return "❌ 벡터스토어가 존재하지 않습니다."
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return "✅ 벡터스토어 로딩 완료!"

def extract_questions(pdf_file):
    global questions, question_index
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
    return f"✅ 총 {len(questions)}개의 문제가 추출되었습니다."

def setup_chain():
    global evaluate_chain, vectorstore
    if vectorstore is None:
        return "❗ 먼저 벡터스토어를 불러오세요."
    retriever = vectorstore.as_retriever()

    # 🚩 PromptTemplate 동일하게 사용 
    prompt = PromptTemplate.from_template("""
너는 영문 독해 시험의 평가자야. 아래 Q(문제)와 A(수험생 답변)가 있을 때,
#Context: (문서에서 뽑은 정보), #Question: (문제), #Answer: (받아쓴 답변)

수험생이 정답에 얼마나 가까운지, 논리적 근거가 정확한지 'context만 활용'해서 한글로 5문장으로 평가, 점수(A~E점, 소수점 가능)로 매겨줘.
피드백은 한글로 부탁해.

#Question: {question}
#Answer: {answer}
#Context: {context}

[출력 예시]   
점수: B점   
피드백: 답변이 핵심 내용을 잘 언급했으나, 세부정보가 빠짐.
""")

    # 🚩 달라지는 부분: OpenAI(4o) 기반 LLM 및 Chain 세팅
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        streaming=False,
    )
    evaluate_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        # context is set at evaluation-time below
    )
    return "✅ 평가 체인 준비 완료!"

def get_next_question():
    global question_index
    if not questions:
        return "❗ 먼저 PDF를 업로드해주세요.", None
    if question_index >= len(questions):
        return "🎉 모든 문제를 완료했습니다!", None

    question = questions[question_index]
    question_index += 1

    # 텍스트, mp3 파일 경로 반환
    tts = gTTS(text=question, lang='en')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    tmp.close()
    return f"[문제 {question_index}] {question}", tmp.name

def evaluate_audio_response(audio):
    global question_index, evaluate_chain
    if evaluate_chain is None:
        return "❌ 평가 체인을 먼저 준비해주세요."
    if question_index == 0:
        return "❗ 문제를 먼저 시작해주세요."

    question = questions[question_index - 1]
    user_answer = speech_to_text(audio)  # audio는 파일경로(str): 이게 파일에서 STT하는 함수여야 함

    # 🚩 context 검색
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # 🚩 gpt-4o로 평가
    result = evaluate_chain.invoke({
        "question": question,
        "answer": user_answer,
        "context": context,   # context 추가
    })
    return f"🧠 AI 평가 결과:\n\n{result['text'] if isinstance(result, dict) and 'text' in result else result}"

with gr.Blocks() as demo:
    gr.Markdown("## 🗣️ 토익스피킹 AI 채점 시스템")
    with gr.Row():
        pdf_input = gr.File(label="📄 문제 PDF 업로드", file_types=[".pdf"])
        upload_result = gr.Textbox(label="문제 추출 결과")
    with gr.Row():
        load_btn = gr.Button("📂 벡터스토어 불러오기")
        vectorstore_status = gr.Textbox(label="벡터스토어 상태")
    with gr.Row():
        setup_btn = gr.Button("🧠 평가 체인 준비")
        chain_status = gr.Textbox(label="평가 체인 상태")

    next_q_btn = gr.Button("📢 다음 문제")
    question_output = gr.Textbox(label="문제 보기")
    question_audio = gr.Audio(label="문제 음성", type="filepath")

    audio_input = gr.Audio(label="🎤 음성 답변", type="filepath")
    result_output = gr.Textbox(label="📊 AI 평가 결과")

    pdf_input.change(fn=extract_questions, inputs=[pdf_input], outputs=[upload_result])
    load_btn.click(fn=load_vectorstore, outputs=[vectorstore_status])
    setup_btn.click(fn=setup_chain, outputs=[chain_status])
    next_q_btn.click(fn=get_next_question, outputs=[question_output, question_audio])
    audio_input.change(fn=evaluate_audio_response, inputs=[audio_input], outputs=[result_output])

demo.launch()