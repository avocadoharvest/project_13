from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from claude_API import build_claude_chain
from example import speech_to_text
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gtts import gTTS
import playsound

def load_vectorstore():
    """저장된 FAISS 벡터스토어를 로드하는 함수"""
    db_path = "./db/faiss"
    
    if not os.path.exists(db_path):
        print("❌ 벡터스토어가 존재하지 않습니다!")
        print("먼저 create_db.py를 실행해서 DB를 생성하세요.")
        return None
    
    print("🔍 저장된 벡터스토어 로딩 중...")
    # 임베딩 모델 (DB 생성할 때와 동일해야 함)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 저장된 벡터스토어 로드
    vectorstore = FAISS.load_local(db_path, embeddings)
    print("✅ 벡터스토어 로딩 완료!")
    
    return vectorstore

def main():
    """메인 실행 함수"""
    
    # 저장된 벡터스토어 로드
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return
    
    retriever = vectorstore.as_retriever()
    
    # RAG 체인용 프롬프트
    prompt = PromptTemplate.from_template("""
    너는 영문 독해 시험의 평가자야. 아래 Q(문제)와 A(수험생 답변)가 있을 때,
    #Context: (문서에서 뽑은 정보), #Question: (문제), #Answer: (받아쓴 답변)

    수험생이 정답에 얼마나 가까운지, 논리적 근거가 정확한지 '문서 내용만 활용'해서 영어로 2~3문장 평가받고, 점수(1~5점, 소수점 가능)로 매겨줘.
    피드백은 한글로 부탁해

    #Question: {question}
    #Answer: {answer}
    #Context: {context}

    [출력 예시]  
    점수: 3.5점  
    피드백: 답변이 핵심 내용을 잘 언급했으나, 세부정보가 빠짐.
    """)
    
    # Claude-RAG 체인 생성
    evaluate_chain = build_claude_chain(retriever, prompt, model_name="claude-3-haiku-20240307", temperature=0)
    
    # 문제 추출
    print("📝 문제 추출 중...")
    question_pdf_path = "toss_part3.pdf"
    loader = PyPDFLoader(question_pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", "?"])
    docs = splitter.split_documents(pages)
    
    questions = []
    for doc in docs:
        lines = doc.page_content.strip().split("\n")
        for line in lines:
            if line.strip().startswith("Q") and "?" in line:
                questions.append(line.strip())
    
    print(f"✅ 총 {len(questions)}개의 문제를 발견했습니다!")
    
    # 문제 루프
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n🎯 [문제 {i}] {question}")
        
        # 문제 읽어주기
        tts = gTTS(text=question, lang='en')
        filename = f"question_{i}.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
        
        # 음성 입력 받기
        user_answer = speech_to_text(15)
        print("🧠 AI 채점 중...")
        
        # AI 평가
        evaluation = evaluate_chain.invoke({"question": question, "answer": user_answer})
        print("💡 AI 평가 결과:\n", evaluation)
        
        # 결과 저장
        results.append({"문제": question, "답변": user_answer, "채점": evaluation})
        input("[엔터] 다음 문제로...")
    
    # 전체 결과 요약
    print("\n===== 전체 결과 요약 =====")
    for i, res in enumerate(results, 1):
        print(f"\n[문제 {i}]: {res['문제']}")
        print(f"🗣️ 답변: {res['답변']}")
        print(f"📊 AI 평가: {res['채점']}")

if __name__ == "__main__":
    main()
