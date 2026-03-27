# modules/clova_embedding.py

import http.client
import json
from langchain.embeddings.base import Embeddings
from typing import List

class ClovaEmbeddingWrapper(Embeddings):
    def __init__(self, api_key: str, request_id: str, host: str = "clovastudio.stream.ntruss.com"):
        self.api_key = api_key
        self.request_id = request_id
        self.host = host

    def _call_clova_embedding(self, text: str) -> List[float]:
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': f'Bearer {self.api_key}',
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id
        }

        body = {
            "text": text
        }

        conn = http.client.HTTPSConnection(self.host)
        conn.request('POST', '/v1/api-tools/embedding/v2', json.dumps(body), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode('utf-8'))
        conn.close()

        if result.get("status", {}).get("code") == "20000":
            return result["result"]["embedding"]
        else:
            raise ValueError(f"임베딩 API 오류: {result}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._call_clova_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._call_clova_embedding(text)