import streamlit as st
import pandas as pd
import joblib

# =========================
# 모델 불러오기
# ======================
# 저장된 모델 파일명 예시: rf_model.pkl
rf_model = joblib.load("tomato_model.pkl")

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="착과율 예측",
    page_icon="🍅",
    layout="centered"
)

# =========================
# 제목
# =========================
st.title("🍅 착과율 예측 시스템")
st.write("환경 데이터를 입력하면 착과율을 예측합니다.")

# =========================
# 사용자 입력
# =========================
st.subheader("환경 정보 입력")

temp = st.number_input(
    "내부온도 (℃)",
    min_value=0.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

humidity = st.number_input(
    "내부습도 (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=0.1
)

soil_temp = st.number_input(
    "지온 (℃)",
    min_value=0.0,
    max_value=50.0,
    value=20.0,
    step=0.1
)

# =========================
# 예측 버튼
# =========================
if st.button("예측하기"):

    # 입력 데이터를 DataFrame으로 변환
    input_data = pd.DataFrame(
        [[temp, humidity, soil_temp]],
        columns=["내부온도", "내부습도", "지온"]
    )

    # 예측 수행
    predicted = rf_model.predict(input_data)

    # 결과 출력
    st.success(f"예측 착과율: {predicted[0]:.1f}%")

    # 입력값 확인용
    st.subheader("입력 데이터")
    st.dataframe(input_data)