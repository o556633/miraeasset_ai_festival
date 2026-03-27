# modules/retriever.py
import os
import pickle
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore


def create_retriever(name: str, embedding_func):
    """
    저장된 ChromaDB와 docstore.pkl을 기반으로 MultiVectorRetriever 생성
    """
    base_path = "./db/DB_"
    persist_path = base_path + name

    # 1. Chroma 불러오기
    vectorstore = Chroma(
        collection_name="summaries",
        embedding_function=embedding_func,
        persist_directory=persist_path
    )

    # 2. Docstore 불러오기
    docstore_path = os.path.join(persist_path, f"db_{name}_docstore.pkl")
    with open(docstore_path, "rb") as f:
        doc_dict = pickle.load(f)

    # 3. dict → InMemoryStore 변환
    store = InMemoryStore()
    store.mset([(k, v) for k, v in doc_dict.items()])

    # 4. MultiVectorRetriever 생성
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id"
    )

    print(f"Retriever 생성 완료: {name}")
    return retriever