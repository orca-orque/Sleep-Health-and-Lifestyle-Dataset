import math

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# 기본 설정 (set_page_config는 스트림릿 명령 중 가장 먼저 실행해야 해요)
# ------------------------------------------------------------------
st.set_page_config(page_title="탐색 - 수면 장애 유무 예측", page_icon="🔎", layout="wide")

# 첫 화면(main.py)과 같은 데이터 주소
DATA_URL = (
    "https://raw.githubusercontent.com/orca-orque/"
    "Sleep-Health-and-Lifestyle-Dataset_data/refs/heads/main/sleep_preprocessed.csv"
)

# 살펴볼 세 가지 값: 열 이름 -> 화면에 보여 줄 이름
FEATURES = {
    "Sleep Duration": "수면 시간 (시간)",
    "Stress Level": "스트레스 수준 (점)",
    "Physical Activity Level": "신체 활동량 (분)",
}
REQUIRED_COLUMNS = ["Gender", "Age", "Sleep Disorder", *FEATURES]

DISORDER_ORDER = ["없음", "있음"]
DISORDER_COLORS = {"없음": "#4C78A8", "있음": "#F58518"}
HIST_COLOR = "#76B7B2"

GENDER_ORDER = ["여성", "남성"]

# 나이를 10세 단위로 묶는 기준 (20 이상 30 미만 = 20대 ...)
AGE_EDGES = [20, 30, 40, 50, 60]
AGE_LABELS = ["20대", "30대", "40대", "50대"]


# ------------------------------------------------------------------
# 데이터 불러오기와 준비
# ------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    # utf-8-sig: UTF-8 파일 맨 앞에 BOM이 있어도 없어도 첫 열 이름이 깨지지 않아요.
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def gender_name(value) -> str:
    """성별이 Male/Female 글자든 0/1 숫자든 '남성'/'여성'으로 바꿔 줘요. (0=남, 1=여)"""
    text = str(value).strip()
    if text in ("0", "Male"):
        return "남성"
    if text in ("1", "Female"):
        return "여성"
    return text


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["수면 장애"] = out["Sleep Disorder"].map({0: "없음", 1: "있음"})
    out["성별"] = out["Gender"].map(gender_name)
    out["나이대"] = pd.cut(
        out["Age"], bins=AGE_EDGES, labels=AGE_LABELS, right=False
    ).astype(str)
    return out


# ------------------------------------------------------------------
# 표와 그래프를 만드는 함수
# ------------------------------------------------------------------
def make_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    groups = [("전체", df)] + [
        (f"수면 장애 {name}", df[df["수면 장애"] == name]) for name in DISORDER_ORDER
    ]
    rows = []
    for title, sub in groups:
        row = {"구분": title, "사람 수": len(sub)}
        for col, label in FEATURES.items():
            row[f"{label} 평균"] = round(float(sub[col].mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def make_ratio_table(df: pd.DataFrame, group_col: str, order: list) -> pd.DataFrame:
    """그룹마다 수면 장애 '없음'/'있음'의 비율(%)을 계산해요. x축 이름에는 사람 수(n)도 적어요."""
    counts = pd.crosstab(df[group_col], df["수면 장애"])
    counts = counts.reindex(index=order, columns=DISORDER_ORDER, fill_value=0)
    totals = counts.sum(axis=1)
    ratio = counts.div(totals.where(totals > 0), axis=0) * 100

    long = ratio.reset_index().melt(
        id_vars=group_col, var_name="수면 장애", value_name="비율(%)"
    )
    long["구분"] = long[group_col].map(lambda g: f"{g} (n={int(totals[g])})")
    return long


def ratio_bar(df: pd.DataFrame, group_col: str, order: list):
    long = make_ratio_table(df, group_col, order)
    x_order = [f"{g} (n={int(df[df[group_col] == g].shape[0])})" for g in order]
    fig = px.bar(
        long,
        x="구분",
        y="비율(%)",
        color="수면 장애",
        barmode="group",
        color_discrete_map=DISORDER_COLORS,
        category_orders={"구분": x_order, "수면 장애": DISORDER_ORDER},
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_yaxes(range=[0, 115], title="비율 (%)")
    fig.update_xaxes(title=None)
    fig.update_layout(legend_title_text="수면 장애")
    return fig


def histogram(df: pd.DataFrame, col: str, label: str):
    fig = px.histogram(
        df, x=col, labels={col: label}, color_discrete_sequence=[HIST_COLOR]
    )
    if col == "Stress Level":
        # 점수가 정수라서 막대 하나가 점수 하나가 되도록 칸을 나눠요.
        start = math.floor(df[col].min()) - 0.5
        end = math.ceil(df[col].max()) + 0.5
        fig.update_traces(xbins=dict(start=start, end=end, size=1))
    else:
        fig.update_traces(nbinsx=12)
    fig.update_layout(yaxis_title="사람 수", bargap=0.05, showlegend=False)
    return fig


def boxplot(df: pd.DataFrame, col: str, label: str):
    fig = px.box(
        df,
        x="수면 장애",
        y=col,
        color="수면 장애",
        color_discrete_map=DISORDER_COLORS,
        category_orders={"수면 장애": DISORDER_ORDER},
        labels={col: label},
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="수면 장애")
    return fig


# ------------------------------------------------------------------
# 화면
# ------------------------------------------------------------------
def main() -> None:
    st.title("🔎 탐색")
    st.caption("수면 시간, 스트레스 수준, 신체 활동량이 어떻게 생겼는지, 수면 장애와 어떤 관계가 있는지 살펴봐요.")

    try:
        raw = load_data()
    except Exception as error:
        st.error("데이터를 불러오지 못했어요. 인터넷 연결과 데이터 주소를 확인해 주세요.")
        st.caption(f"오류 내용: {error}")
        st.stop()

    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        st.error(f"데이터에 없는 열이 있어요: {', '.join(missing)}")
        st.caption(f"지금 데이터의 열: {', '.join(raw.columns)}")
        st.stop()

    df = prepare(raw)

    # 1) 분포: 히스토그램 3개
    st.subheader("1. 값은 어떻게 퍼져 있을까요? (분포)")
    st.caption("막대가 높을수록 그 값을 가진 사람이 많아요.")
    for column, (col, label) in zip(st.columns(3), FEATURES.items()):
        column.markdown(f"**{label}**")
        column.plotly_chart(histogram(df, col, label))

    # 2) 수면 장애 유무별 비교: 상자그림 3개 + 평균 표
    st.subheader("2. 수면 장애가 있는 사람과 없는 사람을 비교해요")
    st.caption("상자 가운데 선은 중앙값, 상자는 가운데 절반의 사람들이 있는 범위예요.")
    for column, (col, label) in zip(st.columns(3), FEATURES.items()):
        column.markdown(f"**{label}**")
        column.plotly_chart(boxplot(df, col, label))

    st.markdown("**평균값 비교**")
    st.dataframe(make_mean_table(df), hide_index=True)

    # 3) 성별별 수면 장애 비율
    st.subheader("3. 성별에 따라 수면 장애 비율이 다를까요?")
    st.caption("각 성별 사람들을 100%로 보았을 때, 수면 장애가 없는 사람과 있는 사람의 비율이에요. n은 사람 수예요.")
    st.plotly_chart(ratio_bar(df, "성별", GENDER_ORDER))

    # 4) 나이대별 수면 장애 비율
    st.subheader("4. 나이에 따라 수면 장애 비율이 다를까요?")
    st.caption("나이를 10세 단위(20대, 30대, 40대, 50대)로 묶어서, 각 나이대 사람들을 100%로 보았을 때의 비율이에요.")
    st.plotly_chart(ratio_bar(df, "나이대", AGE_LABELS))


main()
