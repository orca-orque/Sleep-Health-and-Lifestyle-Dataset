import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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

# '채점표' 페이지와 같은 나누기: 번호 순으로 10명씩 묶고, 묶음마다 앞쪽 3명을 테스트용으로 고정
GROUP_SIZE = 10
TEST_PER_GROUP = 3
RANDOM_STATE = 42

TREE_DEPTH = 5  # 의사결정트리가 던지는 질문 횟수의 최대치
N_SPLITS = 5  # 교차검증에서 사람들을 나누는 덩어리 수
MAX_ROWS = 20  # 목록에 보여 줄 최대 줄 수
UNDEFINED = "계산할 수 없음"

# 채점 결과 네 가지 (수면 장애를 양성으로 둬요)
OUT_TP = "찾아낸 환자 (TP)"
OUT_FN = "놓친 환자 (FN)"
OUT_TN = "아니라고 맞힌 사람 (TN)"
OUT_FP = "헛짚은 사람 (FP)"
OUTCOME_ORDER = [OUT_TP, OUT_FN, OUT_TN, OUT_FP]  # 환자끼리, 아닌 사람끼리 이웃하게 놓아요.

RIGHT_COLOR = "#B0B0B0"  # 맞힌 사람 = 회색
MISS_COLOR = "#D62728"  # 놓친 환자 = 빨강
WRONG_COLOR = "#9467BD"  # 헛짚은 사람 = 보라

MODE_TEST = "test"
MODE_CV = "cv"

# 0.5 경계선: 원래 모델과 성능을 높인 모델
LINE_ORIGINAL = "원래 모델"
LINE_IMPROVED = "성능을 높인 모델"
LINE_COLORS = {LINE_ORIGINAL: "#8C564B", LINE_IMPROVED: "#2CA02C"}  # 원래 = 갈색, 높인 = 초록
POINT_COLORS = {0: "#4C78A8", 1: "#F58518"}  # 실제 수면 장애 없음 = 파랑, 있음 = 주황
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
# 모델 학습과 채점
# ------------------------------------------------------------------
def fit_models(train: pd.DataFrame, features: list) -> dict:
    x_train = train[features]
    y_train = train[TARGET].astype(int)

    # 크기 맞추기(표준화)는 훈련용 사람들로만 계산해요.
    scaler = StandardScaler().fit(x_train)
    logit = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logit.fit(scaler.transform(x_train), y_train)

    # 질문은 다섯 번까지만, 사람이 5명 미만인 마디는 더 나누지 않아요.
    tree = DecisionTreeClassifier(
        max_depth=TREE_DEPTH, min_samples_split=5, random_state=RANDOM_STATE
    )
    tree.fit(x_train, y_train)

    return {"scaler": scaler, "logit": logit, "tree": tree}


def predict_all(models: dict, x: pd.DataFrame) -> dict:
    """(1 = 수면 장애라고 예측)"""
    return {
        NAME_LOGIT: models["logit"].predict(models["scaler"].transform(x)),
        NAME_TREE: models["tree"].predict(x),
    }


def evaluate(df, train, test, features, mode):
    """채점할 사람들과, 그 사람들에 대한 두 모델의 예측을 돌려줘요.

    - 테스트용: 훈련용으로 학습한 모델이 테스트용 사람들을 채점해요.
    - 교차검증: 전체를 N_SPLITS덩어리로 나눠 한 덩어리씩 번갈아 채점해서,
      모든 사람이 자기를 한 번도 보지 않은 모델에게 한 번씩 채점받아요.
    """
    if mode == MODE_TEST:
        people = test.reset_index(drop=True)
        models = fit_models(train, features)
        return people, predict_all(models, people[features])

    people = df.reset_index(drop=True)
    y = people[TARGET].astype(int).to_numpy()
    predictions = {
        NAME_LOGIT: np.zeros(len(people), dtype=int),
        NAME_TREE: np.zeros(len(people), dtype=int),
    }
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fit_idx, eval_idx in splitter.split(people[features], y):
        models = fit_models(people.iloc[fit_idx], features)
        fold_pred = predict_all(models, people.iloc[eval_idx][features])
        for name, pred in fold_pred.items():
            predictions[name][eval_idx] = pred
    return people, predictions


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


def summary_table(people: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """모델(또는 모델 설명) 이름마다 정확도, 재현율, 정밀도, F1과 틀린 사람 수를 한 줄로 보여 줘요."""
    y = people[TARGET].astype(int).to_numpy()
    rows = []
    for name, pred in predictions.items():
        c = confusion_counts(y, pred)
        v = metric_values(c)
        rows.append(
            {
                "모델": name,
                "정확도": percent(v["정확도"]),
                "재현율": percent(v["재현율"]),
                "정밀도": percent(v["정밀도"]),
                "F1": percent(v["F1"]),
                "놓친 환자 (FN)": f"{c['FN']}명",
                "헛짚은 사람 (FP)": f"{c['FP']}명",
            }
        )
    return pd.DataFrame(rows)


def label_outcomes(people: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """사람마다 '찾아낸 환자 / 놓친 환자 / 아니라고 맞힌 사람 / 헛짚은 사람' 중 하나를 붙여요."""
    actual = people[TARGET].astype(int).to_numpy()
    outcome = np.select(
        [(actual == 1) & (pred == 1), (actual == 1) & (pred == 0), (actual == 0) & (pred == 1)],
        [OUT_TP, OUT_FN, OUT_FP],
        default=OUT_TN,
    )
    scored = people.copy()
    scored["결과"] = outcome
    return scored


# ------------------------------------------------------------------
# 목록과 표
# ------------------------------------------------------------------
def person_list(scored: pd.DataFrame, outcome: str):
    """(전체 인원수, 나이가 많은 순서로 정렬한 앞 20줄)"""
    sub = scored[scored["결과"] == outcome].sort_values(["Age", "Person ID"], ascending=[False, True])
    table = pd.DataFrame(
        {
            "번호": sub["Person ID"],
            "하루 평균 수면 시간": sub["Sleep Duration"],
            "하루 신체 활동 시간": sub["Physical Activity Level"],
            "스트레스 수준": sub["Stress Level"],
            "성별": sub["Gender"].map({0: "남성", 1: "여성"}),
            "나이": sub["Age"],
        }
    )
    return len(sub), table.head(MAX_ROWS)


def list_title(label: str, count: int) -> str:
    title = f"{label} (전체 {count}명)"
    if count > MAX_ROWS:
        title += f" · 앞 {MAX_ROWS}줄만 보여요"
    return title


def profile_table(scored: pd.DataFrame) -> pd.DataFrame:
    """네 집단의 평균을 나란히 놓아요."""
    rows = []
    for outcome in OUTCOME_ORDER:
        sub = scored[scored["결과"] == outcome]
        if sub.empty:
            rows.append(
                {
                    "집단": outcome,
                    "사람 수": "0명",
                    "하루 평균 수면 시간": "-",
                    "하루 신체 활동 시간": "-",
                    "스트레스 수준": "-",
                    "나이": "-",
                    "여성 비율": "-",
                }
            )
            continue
        rows.append(
            {
                "집단": outcome,
                "사람 수": f"{len(sub)}명",
                "하루 평균 수면 시간": f"{sub['Sleep Duration'].mean():.2f}시간",
                "하루 신체 활동 시간": f"{sub['Physical Activity Level'].mean():.1f}분",
                "스트레스 수준": f"{sub['Stress Level'].mean():.2f}점",
                "나이": f"{sub['Age'].mean():.1f}세",
                "여성 비율": f"{sub['Gender'].mean() * 100:.0f}%",
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 그림
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


GROUP_OPTIONS = ["나이대", "성별", "스트레스 수준", "하루 평균 수면 시간", "하루 신체 활동 시간"]


def group_column(scored: pd.DataFrame, option: str):
    """사람들을 고른 기준으로 묶어요. (묶음 이름 목록, 묶음 순서)"""
    if option == "나이대":
        order = ["20대", "30대", "40대", "50대"]
        groups = pd.cut(scored["Age"], bins=[20, 30, 40, 50, 60], labels=order, right=False)
    elif option == "성별":
        order = ["남성", "여성"]
        groups = scored["Gender"].map({0: "남성", 1: "여성"})
    elif option == "스트레스 수준":
        values = sorted(scored["Stress Level"].astype(int).unique())
        order = [f"{v}점" for v in values]
        groups = scored["Stress Level"].astype(int).map(lambda v: f"{v}점")
    elif option == "하루 평균 수면 시간":
        order = ["6시간 미만", "6~7시간 미만", "7~8시간 미만", "8시간 이상"]
        groups = pd.cut(scored["Sleep Duration"], bins=[0, 6, 7, 8, 24], labels=order, right=False)
    else:
        order = ["45분 미만", "45~60분 미만", "60~75분 미만", "75분 이상"]
        groups = pd.cut(
            scored["Physical Activity Level"], bins=[0, 45, 60, 75, 1000], labels=order, right=False
        )
    return pd.Series(groups.astype(str).to_numpy()), order


def make_group_figure(scored: pd.DataFrame, option: str):
    """묶음마다 맞힌 사람, 놓친 환자, 헛짚은 사람이 몇 %인지 쌓아서 보여 줘요."""
    groups, order = group_column(scored, option)
    outcome = scored["결과"].to_numpy()
    labels, rights, misses, wrongs = [], [], [], []
    for name in order:
        mask = (groups == name).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        labels.append(f"{name} (n={n})")
        rights.append(float(np.isin(outcome[mask], [OUT_TP, OUT_TN]).mean() * 100))
        misses.append(float((outcome[mask] == OUT_FN).mean() * 100))
        wrongs.append(float((outcome[mask] == OUT_FP).mean() * 100))

    fig = go.Figure()
    for label, values, color in (
        ("맞힌 사람", rights, RIGHT_COLOR),
        (OUT_FN, misses, MISS_COLOR),
        (OUT_FP, wrongs, WRONG_COLOR),
    ):
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                name=label,
                marker_color=color,
                text=[f"{v:.0f}%" if v >= 3 else "" for v in values],
                textposition="inside",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=420,
        margin=dict(t=20),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_yaxes(title="사람 비율 (%)", range=[0, 100])
    fig.update_xaxes(title=option)
    return fig


def make_wrong_scatter(df, scored, x_feat, y_feat):
    """모든 사람을 점으로 찍고, 틀린 사람을 색으로 눈에 띄게 해요."""
    fig = go.Figure()
    for outcomes, label, color, size, opacity in (
        ([OUT_TP, OUT_TN], "맞힌 사람", RIGHT_COLOR, 7, 0.5),
        ([OUT_FN], OUT_FN, MISS_COLOR, 11, 0.85),
        ([OUT_FP], OUT_FP, WRONG_COLOR, 11, 0.85),
    ):
        sub = scored[scored["결과"].isin(outcomes)]
        fig.add_trace(
            go.Scatter(
                x=sub[x_feat],
                y=sub[y_feat],
                mode="markers",
                name=label,
                marker=dict(
                    color=color, size=size, opacity=opacity, line=dict(width=1, color="white")
                ),
                hovertemplate=(
                    f"{FEATURES[x_feat]}: %{{x}}<br>{FEATURES[y_feat]}: %{{y}}<extra>{label}</extra>"
                ),
            )
        )
    fig.update_xaxes(zeroline=False, **axis_layout(df, x_feat))
    fig.update_yaxes(zeroline=False, **axis_layout(df, y_feat))
    fig.update_layout(height=520, legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    return fig


def score_terms(models: dict, model_features: list, axes: list, fixed: dict):
    """점수 = k + (축 값 × 계수의 합). 점수가 0보다 크면 수면 장애일 확률이 0.5보다 커요.

    축이 아닌 속성은 fixed에 적힌 값(중앙값)에 세워 두고 계산해요.
    모델이 쓰지 않는 속성이 축이면 그 계수는 0이라서 선의 모양에 영향을 주지 않아요.
    """
    scaler, logit = models["scaler"], models["logit"]
    mean, scale, coef = scaler.mean_, scaler.scale_, logit.coef_[0]
    k = float(logit.intercept_[0])
    for i, feature in enumerate(model_features):
        if feature not in axes:
            k += coef[i] * (fixed[feature] - mean[i]) / scale[i]
    weights = []
    for feature in axes:
        if feature in model_features:
            i = model_features.index(feature)
            weight = coef[i] / scale[i]
            k -= weight * mean[i]
            weights.append(float(weight))
        else:
            weights.append(0.0)
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


def format_fixed_value(feature: str, value: float) -> str:
    if feature == "Gender":
        if value == 0:
            return "0(남성)"
        if value == 1:
            return "1(여성)"
    return f"{value:g}"


def compare_table(before: dict, after: dict, label_before: str, label_after: str) -> pd.DataFrame:
    """원래 모델과 성능을 높인 모델의 네 지표와 틀린 사람 수를 나란히 놓고, 바뀐 폭도 적어요."""
    mb, ma = metric_values(before), metric_values(after)
    rows = []
    for name in ("정확도", "재현율", "정밀도", "F1"):
        vb, va = mb[name], ma[name]
        diff = "-" if vb is None or va is None else f"{(va - vb) * 100:+.1f}%p"
        rows.append([name, percent(vb), percent(va), diff])
    for label, key in (("놓친 환자 (FN)", "FN"), ("헛짚은 사람 (FP)", "FP")):
        rows.append(
            [label, f"{before[key]}명", f"{after[key]}명", f"{after[key] - before[key]:+d}명"]
        )
    return pd.DataFrame(rows, columns=["항목", label_before, label_after, "바뀐 폭"])


def make_metric_figure(before: dict, after: dict):
    """네 지표를 원래 모델과 성능을 높인 모델로 나란히 막대로 보여 줘요."""
    names = ["정확도", "재현율", "정밀도", "F1"]
    fig = go.Figure()
    for label, counts in ((LINE_ORIGINAL, before), (LINE_IMPROVED, after)):
        values = metric_values(counts)
        heights = [None if values[n] is None else values[n] * 100 for n in names]
        fig.add_trace(
            go.Bar(
                x=names,
                y=heights,
                name=label,
                marker_color=LINE_COLORS[label],
                text=["" if h is None else f"{h:.1f}%" for h in heights],
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        height=340,
        margin=dict(t=20),
        legend=dict(orientation="h", y=-0.25),
    )
    fig.update_yaxes(title="%", range=[0, 112])
    return fig


def make_boundary_figure(df, people, specs, x_feat, y_feat):
    """사람들을 실제 수면 장애 여부로 색을 나눠 찍고, 모델마다 0.5 경계선을 겹쳐 그려요.

    specs: [(이름, 모델 묶음, 그 모델이 쓴 속성 목록), ...]
    """
    fixed = {f: float(people[f].median()) for f in FEATURES}
    x_range, y_range = axis_range(df, x_feat), axis_range(df, y_feat)
    fig = go.Figure()

    for code in (0, 1):
        sub = people[people[TARGET] == code]
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

    sides, notes = {}, []
    for label, models, model_features in specs:
        unused = [
            (axis_name, feature)
            for axis_name, feature in (("가로축", x_feat), ("세로축", y_feat))
            if feature not in model_features
        ]
        if len(unused) == 2:
            notes.append(f"{label}은 두 축 속성을 모두 쓰지 않아서 이 그림에는 선을 그릴 수 없어요.")
            sides[label] = None
            continue
        for axis_name, feature in unused:
            notes.append(
                f"{label}은 '{FEATURES[feature]}' 속성을 쓰지 않아서 선이 {axis_name}과 나란히 그어져요."
            )

        k, (a, c) = score_terms(models, model_features, [x_feat, y_feat], fixed)
        points, side = line_in_box(k, a, c, x_range, y_range)
        sides[label] = side
        if points:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in points],
                    y=[p[1] for p in points],
                    mode="lines",
                    name=f"{label} 0.5 경계선",
                    line=dict(color=LINE_COLORS[label], width=3),
                    hoverinfo="skip",
                )
            )

    fig.update_xaxes(zeroline=False, **axis_layout(df, x_feat))
    fig.update_yaxes(zeroline=False, **axis_layout(df, y_feat))
    fig.update_layout(height=560, legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    return fig, fixed, sides, notes


# ------------------------------------------------------------------
# 화면
# ------------------------------------------------------------------
def show_boundaries(df, train, test, people, predictions, features, mode) -> None:
    st.subheader("5. 원래 모델과 성능을 높인 모델 비교")
    remaining = [f for f in FEATURES if f not in features]
    if not remaining:
        st.info(
            "속성을 다섯 개 모두 골랐어요. 1번에서 속성을 줄이면, 줄인 모델(원래 모델)과 "
            "다섯 개를 모두 쓴 모델(성능을 높인 모델)의 경계선을 비교할 수 있어요."
        )
        return
    extras = st.multiselect(
        "성능을 높이려고 더할 속성",
        options=remaining,
        default=remaining,
        format_func=FEATURES.get,
    )
    if not extras:
        st.info("더할 속성을 하나 이상 골라 주세요.")
        return
    improved = [f for f in FEATURES if f in features or f in extras]

    # 두 모델 모두 훈련용 사람들로 학습해요.
    models_original = fit_models(train, features)
    models_improved = fit_models(train, improved)
    _, predictions_improved = evaluate(df, train, test, improved, mode)

    label_original = f"{LINE_ORIGINAL} (속성 {len(features)}개)"
    label_improved = f"{LINE_IMPROVED} (속성 {len(improved)}개)"
    st.caption(
        f"{LINE_ORIGINAL}은 {', '.join(FEATURES[f] for f in features)}로, "
        f"{LINE_IMPROVED}은 {', '.join(FEATURES[f] for f in improved)}로 학습한 모델이에요. "
        f"아래는 두 모델을 같은 {len(people)}명에게 채점한 결과예요."
    )

    st.markdown("**정확도·재현율·정밀도·F1 비교**")
    y_true = people[TARGET].astype(int).to_numpy()
    left, right = st.columns(2)
    for column, name in zip((left, right), (NAME_LOGIT, NAME_TREE)):
        c_original = confusion_counts(y_true, predictions[name])
        c_improved = confusion_counts(y_true, predictions_improved[name])
        column.markdown(f"**{name}**")
        column.dataframe(
            compare_table(c_original, c_improved, label_original, label_improved),
            hide_index=True,
        )
        column.plotly_chart(make_metric_figure(c_original, c_improved))

    st.markdown(f"**0.5 경계선 비교 ({NAME_LOGIT})**")

    all_features = list(FEATURES)
    c1, c2 = st.columns(2)
    x_default = "Age" if "Age" in all_features else all_features[0]
    x_feat = c1.selectbox(
        "가로축",
        all_features,
        index=all_features.index(x_default),
        format_func=FEATURES.get,
        key="x_line",
    )
    y_options = [f for f in all_features if f != x_feat]
    y_default = "Sleep Duration" if "Sleep Duration" in y_options else y_options[0]
    y_feat = c2.selectbox(
        "세로축",
        y_options,
        index=y_options.index(y_default),
        format_func=FEATURES.get,
        key="y_line",
    )

    specs = [
        (LINE_ORIGINAL, models_original, features),
        (LINE_IMPROVED, models_improved, improved),
    ]
    fig, fixed, sides, notes = make_boundary_figure(df, people, specs, x_feat, y_feat)
    st.plotly_chart(fig)
    st.caption(
        "점은 채점한 사람들이고 주황색은 실제로 수면 장애가 있는 사람이에요. "
        "갈색 선은 원래 모델, 초록색 선은 성능을 높인 모델이 수면 장애(양성)일 확률을 0.5로 가르는 선이에요. "
        "선의 한쪽은 확률이 0.5보다 높은 쪽이에요. "
        f"두 선 모두 훈련용 {len(train)}명으로 학습한 모델의 선이에요."
    )

    others = [f for f in improved if f not in (x_feat, y_feat)]
    if others:
        values = ", ".join(f"{FEATURES[f]} {format_fixed_value(f, fixed[f])}" for f in others)
        st.markdown(f"축이 아닌 속성은 그림에 찍힌 사람들의 중앙값에 세워 두고 계산했어요: **{values}**")
    else:
        st.markdown("두 모델이 쓴 속성을 모두 축으로 썼기 때문에 따로 세워 둔 값은 없어요.")
    for note in notes:
        st.caption(note)
    for label in (LINE_ORIGINAL, LINE_IMPROVED):
        if sides.get(label) is not None:
            st.warning(
                f"{label}의 0.5 경계선은 이 그림 밖에 있어요. "
                f"그림 안에서는 어디서나 수면 장애일 확률이 0.5보다 {sides[label]}."
            )


def main() -> None:
    st.title("🚀 성능 높이기")
    st.caption(
        "모델이 틀린 사람들이 어떤 사람들인지 살펴보면, 모델을 어떻게 고칠지 단서를 찾을 수 있어요. "
        f"수면 장애(1)를 양성으로 두었고, 의사결정트리는 질문을 {TREE_DEPTH}번까지 던져요."
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
        st.info("속성을 두 개 이상 골라 주세요. 그래야 모델을 만들고 틀린 사람을 찾을 수 있어요.")
        st.stop()

    train, test = split_train_test(df)

    # 2) 채점할 사람들 고르기
    st.subheader("2. 어떤 사람들을 채점할까요?")
    mode = st.radio(
        "채점할 사람들",
        [MODE_TEST, MODE_CV],
        horizontal=True,
        format_func=lambda m: (
            f"테스트용 {len(test)}명" if m == MODE_TEST else f"전체 {len(df)}명 (교차검증)"
        ),
    )
    if mode == MODE_TEST:
        st.caption(
            f"훈련용 {len(train)}명으로 학습한 모델이, 처음 보는 테스트용 {len(test)}명을 채점해요. "
            "('채점표' 페이지와 같아요.)"
        )
    else:
        st.caption(
            f"전체 {len(df)}명을 {N_SPLITS}덩어리로 섞어 나눠(수면 장애 비율이 비슷해지게, 난수 고정) "
            "한 덩어리씩 번갈아 채점해요. 모든 사람이 자기를 한 번도 보지 않은 모델에게 한 번씩 채점받고, "
            "크기 맞추기도 덩어리마다 학습용 사람들로만 계산해요. "
            "테스트용보다 틀린 사람을 더 많이 볼 수 있어요."
        )
    people, predictions = evaluate(df, train, test, features, mode)
    st.dataframe(summary_table(people, predictions), hide_index=True)

    # 3) 틀린 사람 목록
    st.subheader("3. 틀린 사람들 목록")
    model_name = st.radio("목록을 볼 모델", [NAME_LOGIT, NAME_TREE], horizontal=True)
    scored = label_outcomes(people, predictions[model_name])
    counts = confusion_counts(people[TARGET].astype(int).to_numpy(), predictions[model_name])
    n_wrong = counts["FN"] + counts["FP"]
    st.markdown(
        f"{len(people)}명 중 틀린 사람은 **{n_wrong}명**({n_wrong / len(people) * 100:.1f}%)이에요. "
        f"놓친 환자 {counts['FN']}명, 헛짚은 사람 {counts['FP']}명이에요."
    )

    missed_count, missed_table = person_list(scored, OUT_FN)
    wrong_count, wrong_table = person_list(scored, OUT_FP)
    left, right = st.columns(2)
    left.markdown(f"**{list_title('놓친 환자: 실제 수면 장애인데 아니라고 예측한 사람', missed_count)}**")
    if missed_count:
        left.dataframe(missed_table, hide_index=True)
    else:
        left.caption("해당하는 사람이 없어요.")
    right.markdown(f"**{list_title('헛짚은 사람: 아닌데 수면 장애라고 예측한 사람', wrong_count)}**")
    if wrong_count:
        right.dataframe(wrong_table, hide_index=True)
    else:
        right.caption("해당하는 사람이 없어요.")

    if missed_count == counts["FN"] and wrong_count == counts["FP"]:
        st.markdown(
            f"왼쪽 목록의 전체 인원 {missed_count}명은 혼동행렬의 FN 칸({counts['FN']}명)과, "
            f"오른쪽 목록의 전체 인원 {wrong_count}명은 FP 칸({counts['FP']}명)과 같아요."
        )
    else:
        st.warning(
            f"목록의 인원(왼쪽 {missed_count}명, 오른쪽 {wrong_count}명)이 "
            f"혼동행렬의 FN({counts['FN']}명), FP({counts['FP']}명)와 달라요."
        )

    # 4) 틀린 사람들은 어떤 사람들일까요?
    st.subheader("4. 틀린 사람들은 어떤 사람들일까요?")

    st.markdown("**네 집단의 평균 비교**")
    st.dataframe(profile_table(scored), hide_index=True)
    st.caption(
        "놓친 환자는 찾아낸 환자와, 헛짚은 사람은 아니라고 맞힌 사람과 비교해 보세요. "
        "평균이 서로 비슷한데도 틀렸다면, 그 사람들은 지금 속성으로는 구분하기 어려운 사람들이에요."
    )

    st.markdown("**어떤 사람들이 많이 틀릴까요?**")
    option = st.selectbox(
        "사람들을 무엇으로 나눠 볼까요?",
        GROUP_OPTIONS,
        help="모델에 쓰지 않은 속성으로 나눠 봐도 돼요. 특정 집단에서 많이 틀린다면 그 속성이 단서일 수 있어요.",
    )
    st.plotly_chart(make_group_figure(scored, option))
    st.caption("막대 하나는 한 집단 사람들을 100%로 보았을 때의 비율이에요. n은 그 집단의 사람 수예요.")

    st.markdown("**틀린 사람들은 그림 어디에 있을까요?**")
    all_features = list(FEATURES)
    c1, c2 = st.columns(2)
    x_feat = c1.selectbox(
        "가로축",
        all_features,
        index=all_features.index(features[0]),
        format_func=FEATURES.get,
        key="x_axis",
    )
    y_options = [f for f in all_features if f != x_feat]
    y_default = features[1] if features[1] in y_options else y_options[0]
    y_feat = c2.selectbox(
        "세로축",
        y_options,
        index=y_options.index(y_default),
        format_func=FEATURES.get,
        key="y_axis",
    )
    st.plotly_chart(make_wrong_scatter(df, scored, x_feat, y_feat))
    st.caption(
        "같은 값을 가진 사람은 점이 겹쳐서 하나처럼 보일 수 있어요. "
        "빨간 점은 놓친 환자, 보라색 점은 헛짚은 사람, 회색 점은 맞힌 사람이에요."
    )

    # 5) 0.5 경계선
    show_boundaries(df, train, test, people, predictions, features, mode)


main()
