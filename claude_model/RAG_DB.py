from dotenv import load_dotenv
load_dotenv()

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_and_save_vectorstore():
    """PDF를 로드하고 FAISS 벡터스토어를 생성해서 저장하는 함수"""
    
    print("📖 PDF 로딩 중...")
    # PDF 로드
    loader = PyMuPDFLoader("toeic.pdf")
    pages = loader.load_and_split()
    
    print("✂️ 문서 분할 중...")
    # 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    
    print("🧠 임베딩 모델 로딩 중...")
    # 임베딩 모델
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("🔍 벡터스토어 생성 중...")
    # FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    # DB 디렉터리 생성
    db_path = "./db/faiss"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    print("💾 벡터스토어 저장 중...")
    # 벡터스토어 저장
    vectorstore.save_local(db_path)
    
    print(f"✅ 벡터스토어가 {db_path}에 성공적으로 저장되었습니다!")
    print(f"📊 총 {len(chunks)}개의 청크가 인덱싱되었습니다.")

if __name__ == "__main__":
    create_and_save_vectorstore()
