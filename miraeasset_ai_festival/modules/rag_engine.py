import os
from modules.config import CLOVA_API_KEY, REQUEST_ID
from modules.chroma_manager import load_pdf_as_documents, create_chroma_and_docstore
from modules.clova_embedding import ClovaEmbeddingWrapper
from modules.retriever import create_retriever
from modules.clova_llm import HyperClovaLLM
from langchain.chains import RetrievalQA
from modules.config import CLOVA_API_KEY, REQUEST_ID

def initialize_rag_qa(folder_path="data", persist_dir="etf"):
    print("RAG QA 시스템 초기화 중...")

    # 이미 persist_dir에 Chroma DB가 존재하면 PDF 로딩 및 재생성 생략
    if not os.path.exists(os.path.join(persist_dir, "index")):
        print("📚 Chroma DB가 존재하지 않아 새로 생성합니다.")
        docs = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                full_path = os.path.join(folder_path, filename)
                print(f"📄 로딩 중: {filename}")
                docs.extend(load_pdf_as_documents(full_path))

        embedding_func = ClovaEmbeddingWrapper(
            api_key=CLOVA_API_KEY,
            request_id=REQUEST_ID
        )
        create_chroma_and_docstore(persist_dir, docs, embedding_func)
    else:
        print("기존 Chroma DB 사용")

    # 공통 처리: retriever + llm + qa 초기화
    embedding_func = ClovaEmbeddingWrapper(
        api_key=CLOVA_API_KEY,
        request_id=REQUEST_ID
    )
    retriever = create_retriever(persist_dir, embedding_func)
    llm = HyperClovaLLM()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    print("RAG QA 시스템 준비 완료.")
    return qa