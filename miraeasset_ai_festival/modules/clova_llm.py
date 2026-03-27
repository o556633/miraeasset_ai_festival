# modules/clova_llm.py
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import BaseModel, Field
import http.client
import json
from modules.config import CLOVA_API_KEY, REQUEST_ID


class HyperClovaLLM(LLM, BaseModel):
    api_key: str = Field(default=CLOVA_API_KEY)
    request_id: str = Field(default=REQUEST_ID)
    host: str = Field(default="clovastudio.stream.ntruss.com")

    @property
    def _llm_type(self) -> str:
        return "hyperclova-x"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id
        }

        body = {
            "messages": [
                {"role": "system", "content": "너는 사용자 포트폴리오 기반으로 최적의 ETF을 제안하는 금융특화 AI서비스야, 사용자 주식과 추천한 ETF의 연관성과 추천 ETF을 보다 자세하게 설명해줘야해."},
                {"role": "user", "content": prompt}
            ],
            "topK": 0,
            "topP": 0.8,
            "temperature": 0.5,
            "maxTokens": 1000,
            "includeAiFilters": True,
            "stopBefore": [],
            "repeatPenalty": 5.0
        }

        conn = http.client.HTTPSConnection(self.host)
        conn.request("POST", "/testapp/v1/chat-completions/HCX-003", json.dumps(body), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode("utf-8"))
        conn.close()

        try:
            return result["result"]["message"]["content"]
        except Exception as e:
            raise ValueError(f"Clova 응답 오류: {result}")
