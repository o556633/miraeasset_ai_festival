# modules/chroma_manager.py

import os
import pickle
import fitz  # PyMuPDF
from typing import List
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_as_documents(pdf_path: str) -> List[Document]:
    """
    PDF 파일을 페이지 단위로 읽어 LangChain Document 리스트로 반환
    """
    doc = fitz.open(pdf_path)
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 50
    )

    for i, page in enumerate(doc):

        text = page.get_text().strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        for j, chunk in enumerate(chunks) :
            documents.append(Document(
                page_content=chunk,
                metadata={"page": i, "doc_id": f"doc_{i}_{j}"}
            ))
    return documents

def create_chroma_and_docstore(name: str, docs: List[Document], embedding_func):
    """
    문서를 임베딩 후 Chroma DB 및 docstore(pkl)로 저장
    """
    base_path = "./db/DB_"
    save_path = base_path + name
    os.makedirs(save_path, exist_ok=True)

    # 1. Chroma DB 저장
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_func,
        collection_name="summaries",
        persist_directory=save_path
    )
    db.persist()

    # 2. Docstore 저장
    doc_dict = {doc.metadata["doc_id"]: doc for doc in docs}
    with open(os.path.join(save_path, f"db_{name}_docstore.pkl"), "wb") as f:
        pickle.dump(doc_dict, f)

    print(f" Chroma DB + Docstore 저장 완료: {save_path}")