# ETF 추천 시스템

이 프로젝트는 투자 결정을 지원하는 **ETF 추천 시스템**입니다.  
사용자의 기존 주식 포트폴리오를 분석하고, 유사도 기반으로 최적 ETF 조합을 추천합니다.  
또한, RAG(Retrieval-Augmented Generation) 기법을 사용해 추천 결과에 대한 해설을 제공합니다.
---
## 주요 기능

- 고객별 주식 포트폴리오 불러오기 및 요약  
- 각 주식과 ETF 벡터 기반 유사도 계산  
- 최적 ETF 조합과 비중 추천
- 추천 ETF 및 포트폴리오 간 유사성 해설 (클로바 X LLM 활용)  
- Streamlit 기반 인터랙티브 웹 앱 제공
---
## 폴더 구조

mirae_asset_ai_festival/
├── main.py                  # 메인 실행 파일 (Streamlit 앱, 프로토타입)
├── requirements.txt         # 필요 라이브러리 목록
├── README.md                # 프로젝트 소개 및 실행법 문서
├── modules/                 # 기능 및 LLM 모델 모듈 (llmrag 및 기타 등)
│   ├── config.py            # API 키 등 민감 정보 설정 파일 (사용자 환경에 맞게 수정 필요)
│   └── ...
├── data/                    # 데이터셋 (CSV, IPython 노트북 등)
│   ├── df_vector.csv        # data_pipeline.ipynb 파일에서 저장되는 주식 벡터값
│   ├── etf_vector.csv       # data_pipeline.ipynb 파일에서 저장되는 ETF 벡터값
│   ├── df_customer.csv      # 분석 목적에 맞게 전처리된 고객별 포트폴리오 데이터
│   ├── customer_dummy.csv   # 고객 더미 데이터 (로우 형식)
│
└── data_pipeline.ipynb      # LLM 모델 활용 제외한 데이터 수집, 가공, 서비스 구현 과정 노트북

---

## 프로젝트 구성 및 역할

- **data_pipeline.ipynb**  
  LLM 모델 연동 전, 데이터 수집부터 전처리, 벡터화, 주식 및 ETF 간 유사도 계산 및 추천 알고리즘 개발까지 수행

- **main.py (Streamlit 앱)**  
  전처리된 데이터와 벡터를 활용해 사용자 인터페이스를 제공하며,  
  `modules` 내 LLM 모듈과 연동해 RAG 기반 추천 해설 기능을 구현

- **modules/config.py**  
  API 키 등 민감 정보를 설정하는 파일로, 사용자 환경에 맞게 반드시 수정 필요

---

## API 키 설정 안내

`modules/config.py` 파일을 열어 아래 변수들을 본인의 클로바 API 키 및 요청 ID로 수정하세요.

```python
CLOVA_API_KEY = "your_clova_api_key_here"
REQUEST_ID = "your_request_id_here"





