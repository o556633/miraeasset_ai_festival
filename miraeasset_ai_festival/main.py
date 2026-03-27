# main.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from modules.rag_engine import initialize_rag_qa
from scipy.optimize import minimize
from itertools import combinations


# ==========================
# 함수 정의
# =====================

def cosine_loss(weights, vectors, target_vector):
    combined = np.average(vectors, axis=0, weights=weights).reshape(1, -1)
    sim = cosine_similarity(combined, target_vector.reshape(1, -1))[0][0]
    return 1 - sim

def recommend_best_etf_combo(etf_dict, portfolio_vector, min_k=1, max_k=4):
    best_score = float('inf')
    best_combo = None
    best_weights = None

    etf_names = list(etf_dict.keys())
    for k in range(min_k, max_k + 1):
        for combo in combinations(etf_names, k):
            vectors = np.array([etf_dict[etf] for etf in combo])
            init_weights = np.ones(k) / k
            bounds = [(0, 1)] * k
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

            result = minimize(
                cosine_loss,
                init_weights,
                args=(vectors, portfolio_vector),
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 200}
            )

            if result.success and result.fun < best_score:
                best_score = result.fun
                best_combo = combo
                best_weights = result.x

    if best_combo is None:
        raise RuntimeError("최적화 실패")

    return best_combo, best_weights, 1 - best_score

def process_customer(customer_id, df_portfolios, stock_vector_dict, etf_vector, top_n=20):
    customer_data = df_portfolios[df_portfolios['customer_id'] == customer_id]
    total_value = customer_data['invested_value'].sum()

    st.subheader(f"👤 고객 ID: {customer_id}")
    st.markdown("### 📊 포트폴리오 요약")
    for _, row in customer_data.iterrows():
        stock = row['stock_name']
        value = row['invested_value']
        pct = value / total_value * 100
        st.write(f"- {stock}: {value:,.0f}원 ({pct:.1f}%)")

    portfolio_vec = np.zeros(len(next(iter(stock_vector_dict.values()))), dtype=float)
    for _, row in customer_data.iterrows():
        stock = row['stock_name']
        val = row['invested_value']
        if stock not in stock_vector_dict:
            st.warning(f"⚠️ '{stock}'는 벡터에 없음 → 무시")
            continue
        vec_data = stock_vector_dict[stock]
        if isinstance(vec_data, dict):
           vec = np.array(list(vec_data.values()), dtype=float)
        else:
           vec = np.array(vec_data, dtype=float)

        portfolio_vec += vec * (val / total_value)


    etf_dict_all = {
        row['ETF_Name']: np.array(row.drop('ETF_Name').values, dtype=float)
        for _, row in etf_vector.iterrows()
    }

    etf_sims = []
    for name, vec in etf_dict_all.items():
        sim = cosine_similarity([vec], [portfolio_vec])[0][0]
        etf_sims.append((name, sim))

    etf_sims.sort(key=lambda x: x[1], reverse=True)
    top_etf_names = [name for name, _ in etf_sims[:top_n]]
    etf_dict = {name: etf_dict_all[name] for name in top_etf_names}

    best_combo, best_weights, best_sim = recommend_best_etf_combo(etf_dict, portfolio_vec)

    st.markdown("### ✅ 최적 추천 ETF 조합")
    for etf in best_combo:
        st.write(f"- {etf}")

    st.markdown("### 📈 포트폴리오 내 ETF 비중")
    for etf, w in zip(best_combo, best_weights):
        st.write(f"{etf}: {w:.4f}")

    st.markdown(f"### 🔗 유사도 점수 (Cosine Similarity): `{best_sim:.4f}`")


    return best_combo, best_weights, portfolio_vec


# =====================
# Streamlit 앱 시작
# =====================

st.title("📈 고객별 ETF 추천 시스템")

# 데이터 불러오기
df_customer = pd.read_csv("data/df_customer.csv")
df_vector = pd.read_csv("data/df_vector.csv")
etf_vector = pd.read_csv("data/etf_vector.csv")

# 벡터 딕셔너리
stock_vector_dict = df_vector.set_index('Name').to_dict(orient='index')


# 고객 선택
customer_ids = df_customer['customer_id'].unique()
selected_customer = st.selectbox("고객을 선택하세요", customer_ids)

if st.button("분석 실행"):
    best_combo, best_weights, _ = process_customer(
        selected_customer, df_customer, stock_vector_dict, etf_vector, top_n=20
    )

    st.markdown("---")
    st.markdown("### 🤖 추천 ETF 해설 (Powered by Clova X)")

    qa = initialize_rag_qa(folder_path="data", persist_dir="etf")

    etf_combo_str = ", ".join(
        [f"{etf} (비중: {weight*100:.1f}%)" for etf, weight in zip(best_combo, best_weights)]
    )

    customer_data = df_customer[df_customer['customer_id'] == selected_customer]
    total_value = customer_data['invested_value'].sum()
    user_stock_str = ", ".join([
        f"{row['stock_name']} (비중: {row['invested_value'] / total_value * 100:.1f}%)"
        for _, row in customer_data.iterrows()
    ])

    query = (
    "다음은 사용자의 주식 포트폴리오 및 추천 ETF 정보입니다.\n\n"
    "1. **사용자 주식 포트폴리오 구성**\n"
    f"- 보유 종목 및 비중: {user_stock_str}\n\n"
    "2. **추천 ETF 조합**\n"
    f"- 추천 ETF 및 비중: {etf_combo_str}\n\n"
    "3. **요청사항**\n"
    "- 위의 사용자 포트폴리오를 기반으로 해당 ETF 조합을 추천한 이유를 자세히 설명해주세요.\n"
    "- 아래 형식을 따라 ETF 4개를 각각 개별적으로 설명해주세요:\n"
    "  1. ETF 이름\n"
    "  2. 주요 편입 종목 및 섹터\n"
    "  3. ETF의 테마 또는 전략\n"
    "  4. 해당 ETF의 최근 시장 상황 또는 수익률 (가능하다면)\n"
    "- 마지막으로, 추천된 각 ETF가 사용자의 기존 포트폴리오와 어떤 **테마, 업종, 스타일 등에서 유사성**을 가지는지, 또는 **보완적인지** 설명해주세요. \n"
    "- 이 분석은 **외부 데이터와 내장된 DB 검색 기반으로** 자세하고 현실적으로 작성해주세요.\n\n"
    "**최종적으로 사용자의 기존 포트폴리오와 추천 ETF 조합이 어떤 점에서 잘 어울리는지, 어떤 투자 전략을 위한 포트폴리오인지 요약해주세요.**"
    )

    try:
        result = qa({"query": query})
        st.markdown("### 🧠 RAG 기반 ETF 추천 해설")
        st.success(result["result"])
    except Exception as e:
        st.error(f"RAG 질의 중 오류 발생: {e}")