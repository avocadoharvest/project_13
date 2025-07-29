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

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from text2speech import speech_to_text  # 파일 경로에서 텍스트로 변환 함수 필요

# 환경변수 및 OpenAI API 키 로딩
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

questions = []
pdf_fulltext = ""
question_index = 0
llm_chain = None      # LLM only (전체 PDF context)
vectorstore = None    # Vector DB
rag_chain = None      # RAG 평가용 (연관 context)

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

def load_vectorstore():
    global vectorstore
    db_path = "./db/faiss"
    if not os.path.exists(db_path):
        return "❌ 벡터스토어가 존재하지 않습니다."
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return "✅ 벡터스토어 로딩 완료!"

def setup_chains():
    global llm_chain, rag_chain, vectorstore
    # (1) PDF 전체 context로 평가하는 LLM체인
    llm_prompt = PromptTemplate.from_template("""
너는 영문 독해 시험의 평가자야. 아래 Q(문제), A(수험생 답변), Context(PDF 전체)가 있어.

#Question: {question}
#Answer: {answer}
#Context: {context}

수험생이 정답에 얼마나 가까운지, 논리적 근거가 정확한지 'context만 활용'해서 한글로 5문장으로 평가, 점수(A~E점, 소수점 가능)로 매겨줘.
피드백은 한글로 부탁해.

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
    llm_chain = LLMChain(
        llm=llm,
        prompt=llm_prompt,
    )
    # (2) RAG-유사도 context(연관 chunk 기반)
    rag_prompt = PromptTemplate.from_template("""
너는 영문 독해 시험의 평가자다. 아래 문제, 답변, context(문서의 일부)가 있다.

#Question: {question}
#Answer: {answer}
#Context: {context}

수험생이 정답에 얼마나 가까운지, 논리적 근거가 정확한지 'context만 활용'해서 한글로 5문장 평가·점수(A~E점, 소수점 허용)로 채점하라.
[출력 예시]
점수: B점
피드백: 답변이 핵심 내용을 잘 언급했으나, 세부정보가 빠짐.
""")
    rag_chain = LLMChain(
        llm=llm,                 # 같은 LLM 인스턴스 사용 (속도상 유리)
        prompt=rag_prompt,
    )
    return "✅ 평가 체인(LMM Only + RAG) 준비 완료!"

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
    global question_index, llm_chain, rag_chain, vectorstore, pdf_fulltext
    if llm_chain is None or rag_chain is None:
        return "❌ 평가 체인을 먼저 준비해주세요.", ""
    if question_index == 0:
        return "❗ 문제를 먼저 시작해주세요.", ""

    question = questions[question_index - 1]
    user_answer = speech_to_text(audio)
    # (1) LLM Only: context는 PDF 전체 본문
    context_all = pdf_fulltext if pdf_fulltext else ""
    llm_result = llm_chain.invoke({
        "question": question,
        "answer": user_answer,
        "context": context_all,
    })
    llm_feedback = f"🟢 [PDF 전체 context 평가]\n{llm_result['text'] if isinstance(llm_result, dict) and 'text' in llm_result else llm_result}"

    # (2) RAG: context는 vectorstore에서 유사 chunk 추출
    if vectorstore is not None:
        docs = vectorstore.similarity_search(question, k=3)
        rag_context = "\n".join([doc.page_content for doc in docs])
        rag_result = rag_chain.invoke({
            "question": question,
            "answer": user_answer,
            "context": rag_context,
        })
        rag_feedback = f"🔵 [RAG(연관 chunk) 평가]\n{rag_result['text'] if isinstance(rag_result, dict) and 'text' in rag_result else rag_result}"
    else:
        rag_feedback = "❗ FAISS 벡터스토어가 적용되지 않았습니다."

    return llm_feedback, rag_feedback

with gr.Blocks() as demo:
    gr.Markdown("## 🗣️ 토익스피킹 AI 채점 시스템 (PDF 전체/벡터 RAG 동시 평가)")

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
    result_output_llm = gr.Textbox(label="📊 [No RAG context 평가]")
    result_output_rag = gr.Textbox(label="📊 [RAG context 평가]")

    pdf_input.change(fn=extract_questions, inputs=[pdf_input], outputs=[upload_result])
    load_btn.click(fn=load_vectorstore, outputs=[vectorstore_status])
    setup_btn.click(fn=setup_chains, outputs=[chain_status])
    next_q_btn.click(fn=get_next_question, outputs=[question_output, question_audio])
    # 오디오 평가 결과 2개로 분리
    audio_input.change(
        fn=evaluate_audio_response,
        inputs=[audio_input],
        outputs=[result_output_llm, result_output_rag]
    )

demo.launch()
