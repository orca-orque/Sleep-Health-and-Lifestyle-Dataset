import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------------------------
# 기본 설정 (set_page_config는 스트림릿 명령 중 가장 먼저 실행해야 해요)
# ------------------------------------------------------------------
st.set_page_config(page_title="성능 높이기 - 수면 장애 유무 예측", page_icon="🚀", layout="wide")

# 첫 화면(main.py), 다른 페이지와 같은 데이터 주소
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

# '채점표', '분류 모델' 페이지와 같은 나누기: 번호 순으로 10명씩 묶고, 묶음마다 앞쪽 3명을 테스트용으로 고정
GROUP_SIZE = 10
TEST_PER_GROUP = 3
RANDOM_STATE = 42

UNDEFINED = "계산할 수 없음"
LABEL_BEFORE = "가중치 맞추기 전"
LABEL_AFTER = "가중치 맞춘 뒤"
BOUNDARY_COLORS = {"before": "#9467BD", "after": "#2CA02C"}  # 전 = 보라, 뒤 = 초록
POINT_COLORS = {0: "#4C78A8", 1: "#F58518"}  # 없음 = 파랑, 있음 = 주황
ANSWER_NAME = {0: "수면 장애 없음", 1: "수면 장애 있음"}


# ------------------------------------------------------------------
# 데이터 불러오기와 나누기 ('채점표' 페이지와 똑같이)
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
# 모델 학습: 가중치를 맞추기 전(balanced=False)과 맞춘 뒤(balanced=True)
# ------------------------------------------------------------------
def fit_models(train: pd.DataFrame, features: list, balanced: bool) -> dict:
    x_train = train[features]
    y_train = train[TARGET].astype(int)
    class_weight = "balanced" if balanced else None

    # 크기 맞추기(표준화)는 훈련용 사람들로만 계산해요.
    scaler = StandardScaler().fit(x_train)
    logit = LogisticRegression(
        max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE
    )
    logit.fit(scaler.transform(x_train), y_train)

    # 질문은 세 번까지만, 사람이 5명 미만인 마디는 더 나누지 않아요.
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=5,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )
    tree.fit(x_train, y_train)

    return {"scaler": scaler, "logit": logit, "tree": tree}


def predict_all(models: dict, x_test: pd.DataFrame) -> dict:
    return {
        NAME_LOGIT: models["logit"].predict(models["scaler"].transform(x_test)),
        NAME_TREE: models["tree"].predict(x_test),
    }


# ------------------------------------------------------------------
# 혼동행렬과 지표 ('채점표' 페이지와 똑같이)
# ------------------------------------------------------------------
def confusion_counts(y_true, y_pred) -> dict:
    """수면 장애(1)를 양성으로 두고 네 칸의 사람 수를 세요."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 행=실제, 열=예측
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def metric_values(c: dict) -> dict:
    """네 칸에서 정확도, 재현율, 정밀도, F1을 구해요. 나눌 수 없으면 None."""
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None  # 분모: 실제 수면 장애인 사람 수
    precision = tp / (tp + fp) if (tp + fp) else None  # 분모: 수면 장애라 예측한 사람 수
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def percent(value) -> str:
    return UNDEFINED if value is None else f"{value * 100:.1f}%"


def before_after_table(before: dict, after: dict) -> pd.DataFrame:
    """네 칸과 네 지표를 전·후로 나란히 놓은 표"""
    mb, ma = metric_values(before), metric_values(after)
    rows = [
        ["찾아낸 환자 (TP)", f"{before['TP']}명", f"{after['TP']}명"],
        ["놓친 환자 (FN)", f"{before['FN']}명", f"{after['FN']}명"],
        ["헛짚은 사람 (FP)", f"{before['FP']}명", f"{after['FP']}명"],
        ["아니라고 맞힌 사람 (TN)", f"{before['TN']}명", f"{after['TN']}명"],
    ]
    for name in ("정확도", "재현율", "정밀도", "F1"):
        rows.append([name, percent(mb[name]), percent(ma[name])])
    return pd.DataFrame(rows, columns=["항목", LABEL_BEFORE, LABEL_AFTER])


def change_line(before: dict, after: dict) -> str:
    return (
        f"찾아낸 환자 {before['TP']}명 → {after['TP']}명 · "
        f"놓친 환자 {before['FN']}명 → {after['FN']}명 · "
        f"헛짚은 사람 {before['FP']}명 → {after['FP']}명"
    )


# ------------------------------------------------------------------
# 0.5 경계선(2차원)과 0.5 평면(3차원) 계산
# ------------------------------------------------------------------
def axis_range(df: pd.DataFrame, feature: str) -> tuple:
    if feature == "Gender":
        return (-0.3, 1.3)
    lo, hi = float(df[feature].min()), float(df[feature].max())
    pad = (hi - lo) * 0.06 or 1.0
    return (lo - pad, hi + pad)


def axis_layout(df: pd.DataFrame, feature: str) -> dict:
    """축 이름은 우리말로, 성별 축은 0/1 대신 남성/여성으로 보여 줘요."""
    layout = {"title": {"text": FEATURES[feature]}, "range": list(axis_range(df, feature))}
    if feature == "Gender":
        layout.update(tickvals=[0, 1], ticktext=["남성", "여성"])
    return layout


def score_terms(models: dict, features: list, axes: list, fixed: dict):
    """점수 = k + (축 값 × 계수의 합). 점수가 0보다 크면 수면 장애일 확률이 0.5보다 커요.

    두 축(또는 세 축)이 아닌 속성은 fixed에 적힌 값(테스트 데이터의 중앙값)에 세워 두고 계산해요.
    """
    scaler, logit = models["scaler"], models["logit"]
    mean, scale, coef = scaler.mean_, scaler.scale_, logit.coef_[0]
    k = float(logit.intercept_[0])
    for i, feature in enumerate(features):
        if feature not in axes:
            k += coef[i] * (fixed[feature] - mean[i]) / scale[i]
    weights = []
    for feature in axes:
        i = features.index(feature)
        weight = coef[i] / scale[i]
        k -= weight * mean[i]
        weights.append(float(weight))
    return float(k), weights


def line_in_box(k, a, c, x_range, y_range):
    """k + a*x + c*y = 0 인 직선을 그림 안에 맞춰 잘라 돌려줘요.

    그림 안에 선이 없으면 (None, '높아요' 또는 '낮아요')를 돌려줘요.
    """
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


def plane_in_box(k, coefs, ranges):
    """k + a*x + b*y + c*z = 0 인 평면을 상자(세 축의 범위) 안에 맞춰 잘라 돌려줘요.

    (꼭짓점 목록, None)을 돌려주고, 상자 안에 평면이 없으면 (None, '높아요' 또는 '낮아요')를 돌려줘요.
    """
    lows = np.array([r[0] for r in ranges], dtype=float)
    spans = np.array([r[1] - r[0] for r in ranges], dtype=float)
    a = np.array(coefs, dtype=float) * spans  # 0~1로 줄인 상자에서의 계수
    k0 = k + float(np.dot(coefs, lows))

    corners = np.array(list(itertools.product([0.0, 1.0], repeat=3)))
    scores = k0 + corners @ a
    if scores.min() >= 0:
        return None, "높아요"
    if scores.max() <= 0:
        return None, "낮아요"

    # 상자의 모서리 12개 가운데 점수의 부호가 바뀌는 곳에서 평면이 지나가요.
    points = []
    for i, j in itertools.combinations(range(8), 2):
        if np.sum(corners[i] != corners[j]) != 1:
            continue
        si, sj = scores[i], scores[j]
        if si * sj < 0:
            points.append(corners[i] + si / (si - sj) * (corners[j] - corners[i]))
        elif si == 0:
            points.append(corners[i])
        elif sj == 0:
            points.append(corners[j])

    unique = []
    for p in points:
        if all(np.abs(p - q).max() > 1e-9 for q in unique):
            unique.append(p)
    if len(unique) < 3:
        return None, "높아요" if scores.min() >= 0 else "낮아요"

    # 꼭짓점을 평면 위에서 한 바퀴 돌아가는 순서로 세워요.
    pts = np.array(unique)
    center = pts.mean(axis=0)
    normal = a / np.linalg.norm(a)
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(normal)))] = 1.0
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    angles = np.arctan2((pts - center) @ e2, (pts - center) @ e1)
    pts = pts[np.argsort(angles)]

    return lows + pts * spans, None


def format_fixed_value(feature: str, value: float) -> str:
    if feature == "Gender":
        if value == 0:
            return "0(남성)"
        if value == 1:
            return "1(여성)"
    return f"{value:g}"


def fixed_values_text(features, axes, fixed) -> str:
    others = [f for f in features if f not in axes]
    if not others:
        return "고른 속성을 모두 축으로 썼기 때문에 따로 세워 둔 값은 없어요."
    values = ", ".join(f"{FEATURES[f]} {format_fixed_value(f, fixed[f])}" for f in others)
    return f"축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{values}**"


# ------------------------------------------------------------------
# 그림
# ------------------------------------------------------------------
def make_scatter_2d(df, test, before, after, features, x_feat, y_feat):
    fixed = {f: float(test[f].median()) for f in features}
    x_range, y_range = axis_range(df, x_feat), axis_range(df, y_feat)
    fig = go.Figure()

    for code in (0, 1):  # 테스트 데이터 점: 실제 수면 장애 여부로 색을 나눠요.
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
                    f"{FEATURES[x_feat]}: %{{x}}<br>{FEATURES[y_feat]}: %{{y}}"
                    f"<extra>실제 {ANSWER_NAME[code]}</extra>"
                ),
            )
        )

    sides = {}
    for key, models, label in (
        ("before", before, LABEL_BEFORE),
        ("after", after, LABEL_AFTER),
    ):
        k, (a, c) = score_terms(models, features, [x_feat, y_feat], fixed)
        points, side = line_in_box(k, a, c, x_range, y_range)
        sides[key] = side
        if points:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in points],
                    y=[p[1] for p in points],
                    mode="lines",
                    name=f"{label} 0.5 경계선",
                    line=dict(color=BOUNDARY_COLORS[key], width=3),
                    hoverinfo="skip",
                )
            )

    fig.update_xaxes(zeroline=False, **axis_layout(df, x_feat))
    fig.update_yaxes(zeroline=False, **axis_layout(df, y_feat))
    fig.update_layout(height=560, legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    return fig, fixed, sides


def make_scatter_3d(df, test, before, after, features, axes):
    fixed = {f: float(test[f].median()) for f in features}
    ranges = [axis_range(df, f) for f in axes]
    fig = go.Figure()

    for code in (0, 1):  # 테스트 데이터 점
        sub = test[test[TARGET] == code]
        fig.add_trace(
            go.Scatter3d(
                x=sub[axes[0]],
                y=sub[axes[1]],
                z=sub[axes[2]],
                mode="markers",
                name=f"실제 {ANSWER_NAME[code]}",
                marker=dict(color=POINT_COLORS[code], size=4, opacity=0.8),
                hovertemplate=(
                    f"{FEATURES[axes[0]]}: %{{x}}<br>{FEATURES[axes[1]]}: %{{y}}"
                    f"<br>{FEATURES[axes[2]]}: %{{z}}<extra>실제 {ANSWER_NAME[code]}</extra>"
                ),
            )
        )

    sides = {}
    for key, models, label in (
        ("before", before, LABEL_BEFORE),
        ("after", after, LABEL_AFTER),
    ):
        k, coefs = score_terms(models, features, axes, fixed)
        vertices, side = plane_in_box(k, coefs, ranges)
        sides[key] = side
        if vertices is not None:
            m = len(vertices)
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=[0] * (m - 2),
                    j=list(range(1, m - 1)),
                    k=list(range(2, m)),
                    color=BOUNDARY_COLORS[key],
                    opacity=0.4,
                    name=f"{label} 0.5 평면",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        height=680,
        margin=dict(t=20, l=0, r=0, b=0),
        legend=dict(orientation="h", y=-0.05),
        scene=dict(
            xaxis=axis_layout(df, axes[0]),
            yaxis=axis_layout(df, axes[1]),
            zaxis=axis_layout(df, axes[2]),
        ),
    )
    return fig, fixed, sides


# ------------------------------------------------------------------
# 화면 조각
# ------------------------------------------------------------------
def show_side_warnings(sides: dict, shape: str) -> None:
    for key, label in (("before", LABEL_BEFORE), ("after", LABEL_AFTER)):
        if sides[key] is not None:
            st.warning(
                f"{label}의 0.5 {shape}은 이 그림 밖에 있어요. "
                f"그림 안에서는 어디서나 수면 장애일 확률이 0.5보다 {sides[key]}."
            )


def show_scatter_2d(df, test, before, after, features) -> None:
    c1, c2 = st.columns(2)
    x_feat = c1.selectbox("가로축", features, index=0, format_func=FEATURES.get, key="x_2d")
    y_options = [f for f in features if f != x_feat]
    y_feat = c2.selectbox("세로축", y_options, index=0, format_func=FEATURES.get, key="y_2d")

    fig, fixed, sides = make_scatter_2d(df, test, before, after, features, x_feat, y_feat)
    st.plotly_chart(fig)
    st.caption(
        "점은 테스트용 사람들이에요. 같은 값을 가진 사람은 점이 겹쳐서 하나처럼 보일 수 있어요. "
        "보라색 선은 가중치를 맞추기 전, 초록색 선은 맞춘 뒤의 로지스틱 회귀(확률로 답하는 모델)가 "
        "수면 장애(양성)일 확률을 0.5로 가르는 선이에요."
    )
    st.markdown(fixed_values_text(features, [x_feat, y_feat], fixed))
    show_side_warnings(sides, "경계선")


def show_scatter_3d(df, test, before, after, features) -> None:
    if len(features) == 3:
        axes = list(features)
    else:
        c1, c2, c3 = st.columns(3)
        x_feat = c1.selectbox("X축", features, index=0, format_func=FEATURES.get, key="x_3d")
        y_options = [f for f in features if f != x_feat]
        y_feat = c2.selectbox("Y축", y_options, index=0, format_func=FEATURES.get, key="y_3d")
        z_options = [f for f in features if f not in (x_feat, y_feat)]
        z_feat = c3.selectbox("Z축", z_options, index=0, format_func=FEATURES.get, key="z_3d")
        axes = [x_feat, y_feat, z_feat]

    fig, fixed, sides = make_scatter_3d(df, test, before, after, features, axes)
    st.plotly_chart(fig)
    st.caption(
        "마우스로 끌면 그림이 돌아가고, 휠로 크기를 바꿀 수 있어요. "
        "보라색 평면은 가중치를 맞추기 전, 초록색 평면은 맞춘 뒤의 로지스틱 회귀(확률로 답하는 모델)가 "
        "수면 장애(양성)일 확률을 0.5로 가르는 면이에요."
    )
    st.markdown(fixed_values_text(features, axes, fixed))
    show_side_warnings(sides, "평면")


# ------------------------------------------------------------------
# 화면
# ------------------------------------------------------------------
def main() -> None:
    st.title("🚀 성능 높이기")
    st.caption(
        "수면 장애가 있는 사람과 없는 사람의 수가 다르면 모델이 많은 쪽에 끌려가기 쉬워요. "
        "양쪽의 가중치를 맞추기 전과 맞춘 뒤의 결과를 나란히 비교해요. 수면 장애(1)를 양성으로 두었어요."
    )

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

    # 1) 입력 속성 고르기 ('분류 모델' 페이지와 같은 자리)
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

    train, test = split_train_test(df)
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int).to_numpy()

    # 가중치를 맞추기 전과 맞춘 뒤의 두 모델 묶음
    before = fit_models(train, features, balanced=False)
    after = fit_models(train, features, balanced=True)
    pred_before = predict_all(before, test[features])
    pred_after = predict_all(after, test[features])

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    w_neg, w_pos = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    st.caption(
        f"훈련용 {len(train)}명 중 수면 장애가 있는 사람은 {n_pos}명, 없는 사람은 {n_neg}명이에요. "
        f"가중치를 맞추면 학습할 때 수면 장애가 있는 사람 한 명을 {w_pos:.2f}명처럼, "
        f"없는 사람 한 명을 {w_neg:.2f}명처럼 세어서 양쪽의 합이 같아져요. "
        f"채점은 테스트용 {len(test)}명을 그대로 한 명씩 세어요."
    )

    # 2) 전·후 비교 표
    st.subheader("2. 가중치를 맞추기 전과 맞춘 뒤")
    left, right = st.columns(2)
    for column, name in zip((left, right), (NAME_LOGIT, NAME_TREE)):
        c_before = confusion_counts(y_test, pred_before[name])
        c_after = confusion_counts(y_test, pred_after[name])
        column.markdown(f"**{name}**")
        column.dataframe(before_after_table(c_before, c_after), hide_index=True)
        column.markdown(change_line(c_before, c_after))

    # 3) 2차원 산점도
    st.subheader("3. 0.5 경계선은 어떻게 움직였을까요? (2차원)")
    show_scatter_2d(df, test, before, after, features)

    # 4) 3차원 산점도
    st.subheader("4. 0.5 평면은 어떻게 움직였을까요? (3차원)")
    if len(features) >= 3:
        show_scatter_3d(df, test, before, after, features)
    else:
        st.caption("속성을 세 개 이상 고르면 3차원 그림도 볼 수 있어요.")


main()
