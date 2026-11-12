import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정 (set_page_config는 스트림릿 명령 중 가장 먼저 실행해야 해요)
# ------------------------------------------------------------------
st.set_page_config(page_title="분류 모델 - 수면 장애 유무 예측", page_icon="🤖", layout="wide")

# 첫 화면(main.py)과 같은 데이터 주소
DATA_URL = (
    "https://raw.githubusercontent.com/orca-orque/"
    "Sleep-Health-and-Lifestyle-Dataset_data/refs/heads/main/sleep_preprocessed.csv"
)

TARGET = "Sleep Disorder"  # 1 = 수면 장애(양성), 0 = 아님

# 입력으로 고를 수 있는 속성: 열 이름 -> 화면에 보여 줄 우리말 이름
FEATURES = {
    "Sleep Duration": "하루 평균 수면 시간",
    "Physical Activity Level": "하루 신체 활동 시간",
    "Stress Level": "스트레스 수준",
    "Gender": "성별",
    "Age": "나이",
}
DEFAULT_FEATURES = ["Sleep Duration", "Physical Activity Level", "Stress Level"]

# 모델 이름 (교과서 이름을 앞에, 쉬운 설명을 괄호 안에)
NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "기준 모델(입력을 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

# 번호 순으로 세워 GROUP_SIZE명씩 묶고, 묶음마다 앞쪽 TEST_PER_GROUP명을 테스트용으로 고정해요.
GROUP_SIZE = 10
TEST_PER_GROUP = 3
RANDOM_STATE = 42

ANSWER_NAME = {0: "수면 장애 없음", 1: "수면 장애 있음"}
POINT_COLORS = {0: "#4C78A8", 1: "#F58518"}  # 없음 = 파랑, 있음 = 주황
LEAF_FILL = {0: "#CFE0F1", 1: "#FCDDBA"}
LINE_COLOR = "#9467BD"

CARD_CSS = """
<style>
.acc-card {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
    background: rgba(128, 128, 128, 0.08);
}
.acc-name { font-size: 0.95rem; opacity: 0.85; min-height: 3.6rem; }
.acc-cap { font-size: 0.8rem; opacity: 0.65; }
.acc-big { font-size: 2.8rem; font-weight: 700; line-height: 1.2; }
.acc-small {
    display: flex; justify-content: space-around;
    font-size: 0.85rem; opacity: 0.85; margin-top: 0.6rem;
}
</style>
"""


# ------------------------------------------------------------------
# 데이터 불러오기와 준비
# ------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    # utf-8-sig: UTF-8 파일 맨 앞에 BOM이 있어도 없어도 첫 열 이름이 깨지지 않아요.
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def gender_code(value) -> float:
    """성별이 Male/Female 글자든 0/1 숫자든 0(남성)/1(여성)로 바꿔 줘요."""
    text = str(value).strip()
    if text in ("0", "Male"):
        return 0
    if text in ("1", "Female"):
        return 1
    return float("nan")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Gender"] = out["Gender"].map(gender_code)
    return out


def split_train_test(df: pd.DataFrame):
    """번호 순으로 정렬해 10명씩 묶고, 묶음마다 앞쪽 3명을 테스트용으로 고정해요.

    마지막 묶음이 10명보다 적으면 그 묶음의 30%를 올림한 만큼만 테스트용으로 둬요.
    """
    ordered = df.sort_values("Person ID").reset_index(drop=True)
    order = pd.Series(np.arange(len(ordered)))
    group = order // GROUP_SIZE
    position = order % GROUP_SIZE
    group_size = group.map(group.value_counts())
    n_test = -(-group_size * TEST_PER_GROUP // GROUP_SIZE)  # 올림
    is_test = (position < n_test).to_numpy()
    train = ordered[~is_test].reset_index(drop=True)
    test = ordered[is_test].reset_index(drop=True)
    return train, test


# ------------------------------------------------------------------
# 모델 만들기
# ------------------------------------------------------------------
def fit_models(train: pd.DataFrame, features: list) -> dict:
    x_train = train[features]
    y_train = train[TARGET].astype(int)

    # 크기 맞추기(표준화)는 훈련용 사람들로만 계산해요.
    scaler = StandardScaler().fit(x_train)
    logit = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logit.fit(scaler.transform(x_train), y_train)

    # 질문은 세 번까지만, 사람이 5명 미만인 마디는 더 나누지 않아요.
    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_split=5, random_state=RANDOM_STATE
    )
    tree.fit(x_train, y_train)

    # 입력을 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델
    base = DummyClassifier(strategy="most_frequent")
    base.fit(x_train, y_train)

    return {"scaler": scaler, "logit": logit, "tree": tree, "base": base}


def predict_logit(models: dict, x: pd.DataFrame) -> np.ndarray:
    return models["logit"].predict(models["scaler"].transform(x))


def accuracies(models: dict, train, test, features) -> list:
    """(모델 이름, 훈련용 정확도, 테스트용 정확도) 목록"""
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)
    x_train, x_test = train[features], test[features]
    return [
        (
            NAME_LOGIT,
            accuracy_score(y_train, predict_logit(models, x_train)),
            accuracy_score(y_test, predict_logit(models, x_test)),
        ),
        (
            NAME_TREE,
            accuracy_score(y_train, models["tree"].predict(x_train)),
            accuracy_score(y_test, models["tree"].predict(x_test)),
        ),
        (
            NAME_BASE,
            accuracy_score(y_train, models["base"].predict(x_train)),
            accuracy_score(y_test, models["base"].predict(x_test)),
        ),
    ]


# ------------------------------------------------------------------
# 그림 1: 산점도 + 로지스틱 회귀 0.5 경계선 + 의사결정트리 칸
# ------------------------------------------------------------------
def axis_range(df: pd.DataFrame, feature: str) -> tuple:
    if feature == "Gender":
        return (-0.3, 1.3)
    lo, hi = float(df[feature].min()), float(df[feature].max())
    pad = (hi - lo) * 0.06 or 1.0
    return (lo - pad, hi + pad)


def logistic_line(models, features, x_feat, y_feat, fixed, x_range, y_range):
    """로지스틱 회귀가 수면 장애 확률을 0.5로 가르는 직선을 그림 안에 맞춰 잘라 돌려줘요.

    그림 안에 선이 없으면 (None, '높아요' 또는 '낮아요')를 돌려줘요.
    """
    scaler, logit = models["scaler"], models["logit"]
    mean, scale = scaler.mean_, scaler.scale_
    coef, intercept = logit.coef_[0], float(logit.intercept_[0])
    ix, iy = features.index(x_feat), features.index(y_feat)

    # 두 축이 아닌 속성은 고정한 값으로 계산해요.
    const = intercept
    for i, feature in enumerate(features):
        if feature not in (x_feat, y_feat):
            const += coef[i] * (fixed[feature] - mean[i]) / scale[i]

    # 점수 = K + A*x + C*y (0보다 크면 수면 장애일 확률이 0.5보다 커요)
    a = coef[ix] / scale[ix]
    c = coef[iy] / scale[iy]
    k = const - a * mean[ix] - c * mean[iy]

    (xmin, xmax), (ymin, ymax) = x_range, y_range
    corners = [k + a * x + c * y for x in (xmin, xmax) for y in (ymin, ymax)]
    if min(corners) >= 0:
        return None, "높아요"
    if max(corners) <= 0:
        return None, "낮아요"

    eps = 1e-12
    points = []
    for x in (xmin, xmax):
        if abs(c) > eps:
            y = -(k + a * x) / c
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                points.append((x, y))
    for y in (ymin, ymax):
        if abs(a) > eps:
            x = -(k + c * y) / a
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                points.append((x, y))

    unique = []
    for p in points:
        if all(abs(p[0] - q[0]) > 1e-9 or abs(p[1] - q[1]) > 1e-9 for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None, "높아요" if min(corners) >= 0 else "낮아요"
    return unique[:2], None


def tree_regions(models, features, x_feat, y_feat, fixed, x_range, y_range, n=160):
    """의사결정트리가 그림을 어떻게 나누는지 격자점마다 답을 구해요."""
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    grid_x, grid_y = np.meshgrid(xs, ys)
    columns = {}
    for feature in features:
        if feature == x_feat:
            columns[feature] = grid_x.ravel()
        elif feature == y_feat:
            columns[feature] = grid_y.ravel()
        else:
            columns[feature] = np.full(grid_x.size, fixed[feature])
    grid = pd.DataFrame(columns)
    z = models["tree"].predict(grid).reshape(grid_x.shape)
    return xs, ys, z


def format_fixed_value(feature: str, value: float) -> str:
    if feature == "Gender":
        if value == 0:
            return "0(남성)"
        if value == 1:
            return "1(여성)"
    return f"{value:g}"


def make_scatter(df, test, models, features, x_feat, y_feat):
    fixed = {f: float(test[f].median()) for f in features}
    x_range, y_range = axis_range(df, x_feat), axis_range(df, y_feat)
    x_label, y_label = FEATURES[x_feat], FEATURES[y_feat]

    fig = go.Figure()

    # 의사결정트리가 나눈 칸 (옅은 색)
    xs, ys, z = tree_regions(models, features, x_feat, y_feat, fixed, x_range, y_range)
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=z,
            zmin=0,
            zmax=1,
            colorscale=[[0, POINT_COLORS[0]], [1, POINT_COLORS[1]]],
            showscale=False,
            opacity=0.22,
            hoverinfo="skip",
        )
    )
    for code in (0, 1):  # 범례용 표시
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(symbol="square", size=14, color=POINT_COLORS[code], opacity=0.35),
                name=f"의사결정트리 칸: {ANSWER_NAME[code]}",
                hoverinfo="skip",
            )
        )

    # 테스트 데이터 점 (실제 수면 장애 여부로 색을 나눠요)
    for code in (0, 1):
        sub = test[test[TARGET] == code]
        fig.add_trace(
            go.Scatter(
                x=sub[x_feat],
                y=sub[y_feat],
                mode="markers",
                name=f"실제 {ANSWER_NAME[code]}",
                marker=dict(
                    color=POINT_COLORS[code],
                    size=9,
                    opacity=0.65,
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=(
                    f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra>실제 {ANSWER_NAME[code]}</extra>"
                ),
            )
        )

    # 로지스틱 회귀 0.5 경계선
    points, side = logistic_line(models, features, x_feat, y_feat, fixed, x_range, y_range)
    if points:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="lines",
                name="로지스틱 회귀 0.5 경계선",
                line=dict(color=LINE_COLOR, width=3),
                hoverinfo="skip",
            )
        )

    fig.update_xaxes(title=x_label, range=list(x_range), zeroline=False)
    fig.update_yaxes(title=y_label, range=list(y_range), zeroline=False)
    if x_feat == "Gender":
        fig.update_xaxes(tickvals=[0, 1], ticktext=["남성", "여성"])
    if y_feat == "Gender":
        fig.update_yaxes(tickvals=[0, 1], ticktext=["남성", "여성"])
    fig.update_layout(height=560, legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    return fig, fixed, side


# ------------------------------------------------------------------
# 그림 2: 의사결정트리 가지 그림 (DOT 문자열)
# ------------------------------------------------------------------
def question_text(feature: str, threshold: float) -> str:
    if feature == "Gender":
        return "성별이 남성인가요?"  # 0=남성, 1=여성이라 '0.5 이하'는 남성이에요.
    return f"{FEATURES[feature]} ≤ {threshold:.2f}?"


def tree_summary(models, features):
    t = models["tree"].tree_
    is_leaf = t.children_left == t.children_right
    answers = [int(np.argmax(t.value[i][0])) for i in range(t.node_count) if is_leaf[i]]
    used = sorted({int(t.feature[i]) for i in range(t.node_count) if not is_leaf[i]})
    return len(answers), answers.count(0), [features[i] for i in used]


def build_tree_dot(models, features, train) -> str:
    tree = models["tree"]
    t = tree.tree_
    y_train = train[TARGET].astype(int).to_numpy()

    # 노드마다 그 자리에 온 훈련용 사람 수와 수면 장애인 사람 수를 직접 세요.
    path = tree.decision_path(train[features])
    totals = np.asarray(path.sum(axis=0)).ravel()
    positives = np.asarray(path.T.dot(y_train)).ravel()

    lines = [
        "digraph Tree {",
        'graph [bgcolor="white", rankdir=TB];',
        'node [shape=box, style="rounded,filled", fontsize=12, color="#888888", fillcolor="white", fontcolor="black"];',
        'edge [fontsize=12, color="#666666", fontcolor="black"];',
    ]
    for i in range(t.node_count):
        n, k = int(totals[i]), int(positives[i])
        stats = f"훈련용 {n}명 중 수면 장애 {k}명 ({k / n * 100:.1f}%)"
        left, right = t.children_left[i], t.children_right[i]
        if left == right:  # 더 묻지 않고 답을 내는 마디
            answer = int(np.argmax(t.value[i][0]))
            label = f"답: {ANSWER_NAME[answer]}\\n{stats}"
            lines.append(f'n{i} [label="{label}", fillcolor="{LEAF_FILL[answer]}"];')
        else:
            question = question_text(features[t.feature[i]], float(t.threshold[i]))
            lines.append(f'n{i} [label="{question}\\n{stats}"];')
            lines.append(f'n{i} -> n{left} [label="예"];')
            lines.append(f'n{i} -> n{right} [label="아니요"];')
    lines.append("}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# 화면 조각
# ------------------------------------------------------------------
def accuracy_card(column, name: str, train_acc: float, test_acc: float) -> None:
    column.markdown(
        f'<div class="acc-card">'
        f'<div class="acc-name">{name}</div>'
        f'<div class="acc-cap">테스트 데이터 정확도</div>'
        f'<div class="acc-big">{test_acc * 100:.1f}%</div>'
        f'<div class="acc-small">'
        f"<span>훈련 데이터 {train_acc * 100:.1f}%</span>"
        f"<span>테스트 데이터 {test_acc * 100:.1f}%</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def show_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    total = len(train) + len(test)
    c1, c2, c3 = st.columns(3)
    c1.metric("전체", f"{total}명")
    c2.metric("훈련용 (학습과 크기 맞추기에 사용)", f"{len(train)}명")
    c3.metric("테스트용 (채점에만 사용)", f"{len(test)}명")

    last = total % GROUP_SIZE or GROUP_SIZE
    text = (
        f"사람을 번호 순으로 세워 {GROUP_SIZE}명씩 묶고, 묶음마다 앞쪽 {TEST_PER_GROUP}명을 "
        "테스트용으로 고정했어요. 나머지 일곱 명으로만 모델을 학습시키고, "
        "크기 맞추기(표준화)도 이 사람들로만 계산해요."
    )
    if last != GROUP_SIZE:
        take = -(-last * TEST_PER_GROUP // GROUP_SIZE)
        text += (
            f" 마지막 묶음은 {last}명뿐이라 30%를 올림한 {take}명만 테스트용으로 뒀어요."
        )
    st.caption(text)


def show_scatter(df, test, models, features) -> None:
    c1, c2 = st.columns(2)
    x_feat = c1.selectbox("가로축", features, index=0, format_func=FEATURES.get, key="x_axis")
    y_options = [f for f in features if f != x_feat]
    y_feat = c2.selectbox("세로축", y_options, index=0, format_func=FEATURES.get, key="y_axis")

    fig, fixed, side = make_scatter(df, test, models, features, x_feat, y_feat)
    st.plotly_chart(fig)

    st.caption(
        "점은 테스트용 사람들이에요. 같은 값을 가진 사람은 점이 겹쳐서 하나처럼 보일 수 있어요. "
        "보라색 선은 로지스틱 회귀(확률로 답하는 모델)가 수면 장애(양성)일 확률을 0.5로 가르는 선이에요. "
        "선의 한쪽은 수면 장애일 확률이 0.5보다 높은 쪽이고, 반대쪽은 낮은 쪽이에요. "
        "옅은 파란색과 주황색 칸은 의사결정트리(질문으로 답하는 모델)가 '없음', '있음'이라고 답하는 칸이에요."
    )

    others = [f for f in features if f not in (x_feat, y_feat)]
    if others:
        values = ", ".join(f"{FEATURES[f]} {format_fixed_value(f, fixed[f])}" for f in others)
        st.markdown(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{values}**")
    else:
        st.markdown("속성을 두 개만 골랐기 때문에 따로 세워 둔 값은 없어요.")

    if side is not None:
        st.warning(
            "로지스틱 회귀가 0.5로 가르는 선은 이 그림 밖에 있어요. "
            f"그림 안에서는 어디서나 수면 장애일 확률이 0.5보다 {side}."
        )


def show_tree(models, features, train) -> None:
    st.graphviz_chart(build_tree_dot(models, features, train))

    n_leaves, n_no, used = tree_summary(models, features)
    st.markdown(
        f"- 답을 내는 마디는 모두 **{n_leaves}칸**이고, 그중 **{n_no}칸**이 '수면 장애 없음'이라고 답해요."
    )
    if used:
        names = ", ".join(FEATURES[f] for f in used)
        st.markdown(f"- 고른 속성 가운데 이 나무가 실제로 물은 것은 **{names}**예요.")
    else:
        st.markdown("- 이 나무는 질문을 하나도 던지지 않고 처음부터 답을 냈어요.")


# ------------------------------------------------------------------
# 화면
# ------------------------------------------------------------------
def main() -> None:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.title("🤖 분류 모델")
    st.caption("수면 장애가 있는지(1) 없는지(0)를 맞히는 두 가지 모델을 만들어 비교해요. 수면 장애를 양성으로 두었어요.")

    try:
        raw = load_data()
    except Exception as error:
        st.error("데이터를 불러오지 못했어요. 인터넷 연결과 데이터 주소를 확인해 주세요.")
        st.caption(f"오류 내용: {error}")
        st.stop()

    needed = ["Person ID", TARGET, *FEATURES]
    missing = [col for col in needed if col not in raw.columns]
    if missing:
        st.error(f"데이터에 없는 열이 있어요: {', '.join(missing)}")
        st.caption(f"지금 데이터의 열: {', '.join(raw.columns)}")
        st.stop()

    df = prepare(raw)
    if df[needed].isna().any().any():
        st.error("데이터에 빈 값이나 알 수 없는 값이 있어요. 성별은 Male/Female 또는 0/1이어야 해요.")
        st.stop()

    # 1) 입력 속성 고르기
    st.subheader("1. 입력으로 쓸 속성 고르기")
    chosen = st.multiselect(
        "모델에게 보여 줄 속성 (두 개 이상)",
        options=list(FEATURES),
        default=DEFAULT_FEATURES,
        format_func=FEATURES.get,
    )
    features = [f for f in FEATURES if f in chosen]  # 항상 같은 순서로 맞춰요.
    if len(features) < 2:
        st.info("속성을 두 개 이상 골라 주세요. 그래야 모델을 만들고 그림을 그릴 수 있어요.")
        st.stop()

    # 2) 훈련용 / 테스트용 나누기
    st.subheader("2. 훈련용과 테스트용으로 나누기")
    train, test = split_train_test(df)
    show_split(train, test)

    models = fit_models(train, features)

    # 3) 정확도 카드
    st.subheader("3. 정확도 비교")
    results = accuracies(models, train, test, features)
    for column, (name, train_acc, test_acc) in zip(st.columns(3), results):
        accuracy_card(column, name, train_acc, test_acc)
    st.caption(
        "정확도는 맞게 답한 사람 수를 전체 사람 수로 나눈 값이에요. "
        "훈련용은 모델이 이미 본 사람들이라, 처음 보는 사람들인 테스트용 정확도가 더 믿을 만해요."
    )

    # 4) 산점도
    st.subheader("4. 두 모델은 그림을 어떻게 나눌까요?")
    show_scatter(df, test, models, features)

    # 5) 의사결정트리 가지 그림
    st.subheader("5. 의사결정트리가 던진 질문")
    show_tree(models, features, train)


main()
