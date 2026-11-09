
import streamlit as st
import os
from PIL import Image

st.set_page_config(
    page_title="AI 분리수거 도우미",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ AI 분리수거 도우미")
st.write("쓰레기 이미지를 선택하거나 직접 업로드하여 분리배출 방법을 확인하세요.")

# 분류할 이미지 후보
labels = {
    "a plastic water bottle": "플라스틱",
    "an aluminum drink can": "캔류",
    "a paper cup": "종이컵류",
    "a glass bottle": "유리류",
    "a plastic shopping bag": "비닐류",
    "a cardboard box": "종이류",
    "food scraps": "음식물류"
}

categories = list(set(labels.values()))

guides = {
    "플라스틱": "내용물을 비우고 헹군 뒤 배출하세요.",
    "캔류": "내용물을 비우고 가능한 한 압착하여 배출하세요.",
    "종이컵류": "내용물을 비우고 헹군 뒤 지역별 배출 기준을 확인하세요.",
    "유리류": "내용물을 비우고 깨끗하게 배출하세요.",
    "비닐류": "이물질을 제거하고 깨끗한 비닐만 배출하세요.",
    "종이류": "테이프와 이물질을 제거하고 배출하세요.",
    "음식물류": "음식물 쓰레기 배출 기준에 따라 배출하세요."
}

# 오류 기록
if "errors" not in st.session_state:
    st.session_state.errors = []

# AI 모델 불러오기
@st.cache_resource
def load_model():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )

st.divider()

# 이미지 업로드 기능
st.subheader("📷 직접 이미지 업로드")

uploaded_file = st.file_uploader(
    "컴퓨터에서 쓰레기 이미지를 선택하세요.",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="업로드한 이미지", width=300)

    if st.button("AI 이미지 분석하기"):
        try:
            with st.spinner("AI가 이미지를 분석하고 있습니다..."):
                model = load_model()
                result = model(
                    image,
                    candidate_labels=list(labels.keys())
                )

                best = result[0]
                predicted = labels[best["label"]]

                st.session_state.upload_result = {
                    "name": uploaded_file.name,
                    "prediction": predicted,
                    "score": best["score"]
                }

        except Exception as e:
            st.error("이미지 분석에 실패했습니다.")
            st.caption(str(e))

    if (
        "upload_result" in st.session_state
        and st.session_state.upload_result["name"] == uploaded_file.name
    ):
        result = st.session_state.upload_result
        st.divider()
        st.subheader("🤖 AI 분석 결과")

        st.success(f"예측 결과: {result['prediction']}")
        st.write(
            f"모델 유사도 점수: {result['score']:.2%}"
        )
        st.caption(
            "유사도 점수는 실제 분류 정확도와 같지 않습니다."
        )

        if result["score"] < 0.20:
            st.warning("분류 결과가 불확실할 수 있습니다. 직접 확인해 주세요.")

        st.write("### 분리배출 방법")
        st.write(guides[result["prediction"]])

        st.write("### 정답 확인")
        answer = st.selectbox(
            "실제 쓰레기 종류를 선택하세요.",
            categories,
            key="upload_answer"
        )

        if st.button("정답 확인 및 오류 기록"):
            wrong = result["prediction"] != answer

            st.session_state.errors.append({
                "이미지": result["name"],
                "AI 예측": result["prediction"],
                "실제 정답": answer,
                "오류 여부": wrong
            })

            if wrong:
                st.warning("AI의 오인식 사례가 기록되었습니다.")
            else:
                st.success("정확하게 분류했습니다!")

st.divider()

# 기본 제공 이미지 선택 기능
st.subheader("🗂️ 준비된 이미지 선택")

items = {
    "플라스틱 물병": ("🧴", "플라스틱"),
    "알루미늄 캔": ("🥫", "캔류"),
    "종이컵": ("🥤", "종이컵류"),
    "유리병": ("🍾", "유리류"),
    "비닐봉지": ("🛍️", "비닐류"),
    "종이 상자": ("📦", "종이류")
}

cols = st.columns(3)

for i, (name, data) in enumerate(items.items()):
    emoji, category = data

    with cols[i % 3]:
        st.markdown(f"## {emoji}")
        st.write(name)

        if st.button("선택하기", key=name):
            st.session_state.selected = name

if st.session_state.get("selected"):
    name = st.session_state.selected
    emoji, correct = items[name]

    # 기존 초기 버전의 테스트용 오인식
    mistakes = {
        "종이컵": "일반 종이",
        "유리병": "플라스틱",
        "비닐봉지": "플라스틱"
    }

    prediction = mistakes.get(name, correct)

    st.subheader("기본 이미지 분석 결과")
    st.write(f"선택한 물건: {name}")
    st.info(f"예측 결과: {prediction}")

    if prediction != correct:
        st.warning("오인식이 발생했습니다.")
    else:
        st.success("올바르게 분류했습니다.")

    st.write(guides[correct])

    if st.button("기본 이미지 결과 기록"):
        st.session_state.errors.append({
            "이미지": name,
            "AI 예측": prediction,
            "실제 정답": correct,
            "오류 여부": prediction != correct
        })
        st.success("분류 결과를 기록했습니다.")

st.divider()

# 오류 통계
st.subheader("📊 오류 분석 통계")

total = len(st.session_state.errors)
wrong = sum(
    1 for x in st.session_state.errors if x["오류 여부"]
)

col1, col2 = st.columns(2)
col1.metric("분석 기록", f"{total}개")
col2.metric("오인식 횟수", f"{wrong}개")

if total > 0:
    st.dataframe(
        st.session_state.errors,
        use_container_width=True
    )

    if st.button("오류 기록 초기화"):
        st.session_state.errors = []
        st.rerun()
else:
    st.write("아직 기록된 결과가 없습니다.")
import streamlit as st
from PIL import Image

st.title("♻️ AI 분리수거 도우미")
st.write("이미지를 선택하고 올바른 분리수거 방법을 확인해 보세요.")

categories = ["종이", "플라스틱", "유리", "캔", "일반 쓰레기"]

# 이미지 업로드
st.subheader("📷 이미지 업로드")

uploaded_file = st.file_uploader(
    "분리수거할 이미지 파일을 선택하세요.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", width=300)

    st.write("이 물건은 어떤 종류인가요?")

    selected_category = st.selectbox(
        "분리수거 종류 선택",
        categories
    )

    if st.button("분리수거 결과 확인"):
        st.success(f"선택한 분류: {selected_category}")

        disposal_info = {
            "종이": "이물질을 제거하고 종이류로 배출하세요.",
            "플라스틱": "내용물을 비우고 깨끗하게 헹군 뒤 배출하세요.",
            "유리": "내용물을 비우고 유리류로 배출하세요.",
            "캔": "내용물을 비우고 가능한 한 압착해 배출하세요.",
            "일반 쓰레기": "재활용이 어려운 경우 종량제 봉투에 배출하세요."
        }

        st.info(disposal_info[selected_category])
