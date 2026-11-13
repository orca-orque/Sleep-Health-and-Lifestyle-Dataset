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

# 원래(기본 속성 기준) 모델 성능 지표 고정값
BASELINE_METRICS = {
    NAME_LOGIT: {
        "정확도": "67.3%",
        "재현율": "54.8%",
        "정밀도": "56.1%",
        "F1": "55.4%",
    },
    NAME_TREE: {
        "정확도": "87.6%",
        "재현율": "90.5%",
        "정밀도": "79.2%",
        "F1": "84.4%",
    }
}

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


# ------------------------------------------------------------------
# 데이터 불러오기와 나누기 ('채점표' 페이지와 똑같이)
# ------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def gender_code(value) -> float:
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

    scaler = StandardScaler().fit(x_train)
    logit = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logit.fit(scaler.transform(x_train), y_train)

    tree = DecisionTreeClassifier(
        max_depth=TREE_DEPTH, min_samples_split=5, random_state=RANDOM_STATE
    )
    tree.fit(x_train, y_train)

    return {"scaler": scaler, "logit": logit, "tree": tree}


def predict_all(models: dict, x: pd.DataFrame) -> dict:
    return {
        NAME_LOGIT: models["logit"].predict(models["scaler"].transform(x)),
        NAME_TREE: models["tree"].predict(x),
    }


def evaluate(df, train, test, features, mode):
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
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def metric_values(c: dict) -> dict:
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def percent(value) -> str:
    return UNDEFINED if value is None else f"{value * 100:.1f}%"


def summary_table(people: pd.DataFrame, predictions: dict) -> pd.DataFrame:
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


def comparison_table(people: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """원래 모델 성능과 현재 모델 성능을 한눈에 비교하는 표 생성"""
    y = people[TARGET].astype(int).to_numpy()
    rows = []
    for name, pred in predictions.items():
        c = confusion_counts(y, pred)
        v = metric_values(c)
        
        # 원래 모델 지표 가져오기
        base = BASELINE_METRICS.get(name, {})
        
        rows.append(
            {
                "구분": "원래 모델 (기본 속성)",
                "모델": name,
                "정확도": base.get("정확도", "-"),
                "재현율": base.get("재현율", "-"),
                "정밀도": base.get("정밀도", "-"),
                "F1": base.get("F1", "-"),
            }
        )
        rows.append(
            {
                "구분": "현재 모델 (선택 속성)",
                "모델": name,
                "정확도": percent(v["정확도"]),
                "재현율": percent(v["재현율"]),
                "정밀도": percent(v["정밀도"]),
                "F1": percent(v["F1"]),
            }
        )
    return pd.DataFrame(rows)


def label_outcomes(people: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
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


def axis_range(df: pd.DataFrame, feature: str) -> tuple:
    if feature == "Gender":
        return (-0.3, 1.3)
    lo, hi = float(df[feature].min()), float(df[feature].max())
    pad = (hi - lo) * 0.06 or 1.0
    return (lo - pad, hi + pad)


def axis_layout(df: pd.DataFrame, feature: str) -> dict:
    layout = {"title": {"text": FEATURES[feature]}, "range": list(axis_range(df, feature))}
    if feature == "Gender":
        layout.update(tickvals=[0, 1], ticktext=["남성", "여성"])
    return layout


GROUP_OPTIONS = ["나이대", "성별", "스트레스 수준", "하루 평균 수면 시간", "하루 신체 활동 시간"]


def group_column(scored: pd.DataFrame, option: str):
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


def make_wrong_scatter_with_boundary(df, scored, x_feat, y_feat, features, train_df):
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

    x_min, x_max = axis_range(df, x_feat)
    y_min, y_max = axis_range(df, y_feat)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))

    def add_boundary(feat_list, line_color, line_name, dash_style):
        if x_feat not in feat_list or y_feat not in feat_list:
            return
        
        models = fit_models(train_df, feat_list)
        scaler = models["scaler"]
        logit = models["logit"]

        grid_data = pd.DataFrame(index=np.arange(xx.ravel().shape[0]))
        for f in feat_list:
            if f == x_feat:
                grid_data[f] = xx.ravel()
            elif f == y_feat:
                grid_data[f] = yy.ravel()
            else:
                grid_data[f] = df[f].mean()

        try:
            probs = logit.predict_proba(scaler.transform(grid_data[feat_list]))[:, 1]
            zz = probs.reshape(xx.shape)

            fig.add_trace(
                go.Contour(
                    z=zz,
                    x=np.linspace(x_min, x_max, 150),
                    y=np.linspace(y_min, y_max, 150),
                    contours=dict(start=0.5, end=0.5, size=0, coloring="none"),
                    line=dict(color=line_color, width=3, dash=dash_style),
                    name=line_name,
                    showscale=False,
                )
            )
        except Exception:
            pass

    add_boundary(DEFAULT_FEATURES, "#1f77b4", "원래 0.5 경계선 (기본속성)", "dash")
    
    if set(features) != set(DEFAULT_FEATURES):
        add_boundary(features, "#2ca02c", "성능 높인 0.5 경계선 (선택속성)", "solid")

    fig.update_xaxes(zeroline=False, **axis_layout(df, x_feat))
    fig.update_yaxes(zeroline=False, **axis_layout(df, y_feat))
    fig.update_layout(height=520, legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    return fig


# ------------------------------------------------------------------
# 화면
# ------------------------------------------------------------------
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

    # 1) 입력 속성 고르기
    st.subheader("1. 입력으로 쓸 속성 고르기")
    chosen = st.multiselect(
        "모델에게 보여 줄 속성 (두 개 이상)",
        options=list(FEATURES),
        default=DEFAULT_FEATURES,
        format_func=FEATURES.get,
    )
    features = [f for f in FEATURES if f in chosen]
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
            "크기 맞추기도 덩어리마다 학습용 사람들로만 계산해요."
        )
    
    people, predictions = evaluate(df, train, test, features, mode)
    
    # 기본 요약 표 출력
    st.markdown("#### 📌 현재 설정 모델 성능 요약")
    st.dataframe(summary_table(people, predictions), hide_index=True)

    # 원래 모델 vs 현재 모델 성능 비교 표 추가
    st.markdown("#### 📊 원래 모델(기본 속성) vs 현재 모델(선택 속성) 비교")
    st.caption("기존 원래 모델의 성능(로지스틱 회귀: 정확도 67.3% / 의사결정트리: 정확도 87.6%)과 현재 선택한 속성으로 학습한 모델 성능을 비교합니다.")
    st.dataframe(comparison_table(people, predictions), hide_index=True)

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

    st.markdown("**틀린 사람들은 그림 어디에 있을까요? (0.5 경계선 비교)**")
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
    
    st.plotly_chart(make_wrong_scatter_with_boundary(df, scored, x_feat, y_feat, features, train))
    st.caption(
        "파란색 점선은 '원래 0.5 경계선(기본 속성)'이며, 초록색 실선은 '성능을 높인 0.5 경계선(선택한 속성)'입니다. "
        "속성이 같으면 하나의 경계선만 표시됩니다."
    )


main()
