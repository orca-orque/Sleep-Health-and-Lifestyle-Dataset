import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# 기본 설정 (set_page_config는 스트림릿 명령 중 가장 먼저 실행해야 해요)
# ------------------------------------------------------------------
APP_TITLE = "수면 장애 유무 예측"
APP_ICON = "😴"
DATA_URL = (
    "https://raw.githubusercontent.com/orca-orque/"
    "Sleep-Health-and-Lifestyle-Dataset_data/refs/heads/main/sleep_preprocessed.csv"
)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# 열 이름 -> 우리말 뜻
COLUMN_MEANING = {
    "Person ID": "사람 번호 (한 사람마다 다른 고유 번호)",
    "Gender": "성별 (0=남, 1=여)",
    "Age": "나이 (세)",
    "Sleep Duration": "하루 평균 수면 시간 (시간)",
    "Physical Activity Level": "하루 신체 활동 시간 (분)",
    "Stress Level": "스트레스 수준 (1~10점 척도, 높을수록 스트레스가 큼)",
    "Sleep Disorder": "수면 장애 유무 (0=없음, 1=있음)",
}

# 값 -> 화면에 보여 줄 이름
VALUE_LABELS = {
    "Gender": {"Male": "Male(남)", "Female": "Female(여)", 0: "0(남)", 1: "1(여)"},
    "Sleep Disorder": {0: "0(없음)", 1: "1(있음)"},
}

CARD_CSS = """
<style>
.stat-card {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    background: rgba(128, 128, 128, 0.08);
}
.stat-label {
    font-size: 0.95rem;
    opacity: 0.75;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.2;
}
</style>
"""


# ------------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    # utf-8-sig: 파일 맨 앞에 BOM이 있어도 없어도 첫 열 이름이 깨지지 않아요.
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def value_kind(name: str, series: pd.Series) -> str:
    """열에 들어 있는 값의 종류를 한 줄로 설명해요."""
    values = series.dropna()
    if values.empty:
        return "값 없음"

    is_numeric = pd.api.types.is_numeric_dtype(values)
    n_unique = values.nunique()

    if is_numeric and n_unique == len(values):
        return f"고유 번호 ({values.min():g} ~ {values.max():g})"

    if n_unique <= 2 or not is_numeric:
        labels = VALUE_LABELS.get(name, {})
        if n_unique <= 10:
            shown = " / ".join(labels.get(v, str(v)) for v in sorted(values.unique()))
            return f"{n_unique}가지: {shown}"
        return f"{n_unique}가지"

    return f"숫자 ({values.min():g} ~ {values.max():g})"


def make_column_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "열 이름": col,
                "우리말 뜻": COLUMN_MEANING.get(col, "-"),
                "값의 종류": value_kind(col, df[col]),
                "빈 값 개수": int(df[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def stat_card(column, label: str, value: str) -> None:
    column.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# 화면: 데이터 소개
# ------------------------------------------------------------------
def show_intro(df: pd.DataFrame) -> None:
    st.markdown(
        "이 데이터는 사람들의 **수면·생활 습관 조사 자료**(Sleep Health and Lifestyle)를 "
        "이 앱에 맞게 정리한 표예요. 한 줄이 한 사람이고, 그 사람의 성별, 나이, 수면 시간, "
        "신체 활동량, 스트레스 수준, 수면 장애 여부가 담겨 있어요. "
        "원래 수면 장애는 '없음 / 불면증 / 수면무호흡'의 세 가지였는데, "
        "'있음'과 '없음' 두 가지로 합쳐서 **없음=0, 있음=1**로 바꿨어요. "
        "이 데이터로 수면 장애가 있는지 없는지를 예측하는 모델을 만들어 볼 거예요."
    )

    # 큰 숫자 카드 4개
    n_people = len(df)
    n_columns = len(df.columns)
    n_disorder = int((df["Sleep Disorder"] == 1).sum())
    ratio = n_disorder / n_people * 100 if n_people else 0.0

    st.subheader("데이터 한눈에 보기")
    c1, c2, c3, c4 = st.columns(4)
    stat_card(c1, "전체 사람 수", f"{n_people:,}명")
    stat_card(c2, "열 개수", f"{n_columns}개")
    stat_card(c3, "수면 장애가 있는 사람 (Sleep Disorder = 1)", f"{n_disorder:,}명")
    stat_card(c4, "수면 장애가 있는 사람의 비율", f"{ratio:.1f}%")

    # 열 설명 표
    st.subheader("열 설명")
    st.dataframe(make_column_table(df), hide_index=True)


# ------------------------------------------------------------------
# 화면 맨 아래: 데이터 출처
# ------------------------------------------------------------------
def show_source() -> None:
    st.divider()
    st.subheader("데이터 출처")
    st.markdown(
        "- **게시자**: Adil Shamim\n"
        "- **라이선스**: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)\n"
        "- **사사(출처 표기)**: This dataset is based on a public sample dataset from "
        "Gigasheet and is shared for educational and research purposes"
    )


# ------------------------------------------------------------------
# 앱 실행
# ------------------------------------------------------------------
def main() -> None:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.title(f"{APP_ICON} {APP_TITLE}")

    try:
        df = load_data()
    except Exception as error:
        st.error("데이터를 불러오지 못했어요. 인터넷 연결과 데이터 주소를 확인해 주세요.")
        st.caption(f"오류 내용: {error}")
        st.stop()

    missing = [col for col in COLUMN_MEANING if col not in df.columns]
    if missing:
        st.error(f"데이터에 없는 열이 있어요: {', '.join(missing)}")
        st.caption(f"지금 데이터의 열: {', '.join(df.columns)}")
        st.stop()

    show_intro(df)
    show_source()


main()
