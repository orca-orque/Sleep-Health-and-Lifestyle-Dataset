import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정 (set_page_config는 스트림릿 명령 중 가장 먼저 실행해야 해요)
# ------------------------------------------------------------------
st.set_page_config(page_title="채점표 - 수면 장애 유무 예측", page_icon="📝", layout="wide")

# 첫 화면(main.py), '분류 모델' 페이지와 같은 데이터 주소
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
NAME_BASE = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"

# '분류 모델' 페이지와 같은 나누기: 번호 순으로 10명씩 묶고, 묶음마다 앞쪽 3명을 테스트용으로 고정
GROUP_SIZE = 10
TEST_PER_GROUP = 3
RANDOM_STATE = 42

MAX_ROWS = 20  # 목록에 보여 줄 최대 줄 수
UNDEFINED = "계산할 수 없음"


# ------------------------------------------------------------------
# 데이터 불러오기, 나누기, 모델 학습 ('분류 모델' 페이지와 똑같이)
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


def predict_all(models: dict, x_test: pd.DataFrame) -> dict:
    """테스트용 사람들에 대한 세 모델의 예측 (1 = 수면 장애라고 예측)"""
    return {
        NAME_LOGIT: models["logit"].predict(models["scaler"].transform(x_test)),
        NAME_TREE: models["tree"].predict(x_test),
        NAME_BASE: models["base"].predict(x_test),
    }


# ------------------------------------------------------------------
# 혼동행렬과 지표
# ------------------------------------------------------------------
def confusion_counts(y_true, y_pred) -> dict:
    """수면 장애(1)를 양성으로 두고 네 칸의 사람 수를 세요."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 행=실제, 열=예측
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def confusion_table(c: dict) -> pd.DataFrame:
    """행은 실제, 열은 예측. 칸마다 약자를 함께 적고 행과 열의 합계도 적어요."""
    actual_pos = c["TP"] + c["FN"]
    actual_neg = c["FP"] + c["TN"]
    pred_pos = c["TP"] + c["FP"]
    pred_neg = c["FN"] + c["TN"]
    total = actual_pos + actual_neg
    return pd.DataFrame(
        [
            ["실제: 수면 장애", f"TP · {c['TP']}명", f"FN · {c['FN']}명", f"{actual_pos}명"],
            ["실제: 아님", f"FP · {c['FP']}명", f"TN · {c['TN']}명", f"{actual_neg}명"],
            ["합계", f"{pred_pos}명", f"{pred_neg}명", f"{total}명"],
        ],
        columns=["실제 ↓ / 예측 →", "예측: 수면 장애", "예측: 아님", "합계"],
    )


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


# ------------------------------------------------------------------
# 놓친 환자 / 헛짚은 사람 목록
# ------------------------------------------------------------------
def person_list(test: pd.DataFrame, mask: np.ndarray):
    """(전체 인원수, 나이가 많은 순서로 정렬한 앞 20줄)"""
    sub = test[mask].sort_values(["Age", "Person ID"], ascending=[False, True])
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


# ------------------------------------------------------------------
# 화면
# ------------------------------------------------------------------
def main() -> None:
    st.title("📝 채점표")
    st.caption(
        "테스트용 사람들을 두 모델이 맞혔는지 틀렸는지 표로 채점해요. "
        "수면 장애(1)를 양성으로 두었어요."
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

    # 입력 속성 (분류 모델 페이지와 같은 기본값)
    st.subheader("1. 모델에 쓴 속성")
    chosen = st.multiselect(
        "모델에게 보여 줄 속성 (두 개 이상)",
        options=list(FEATURES),
        default=DEFAULT_FEATURES,
        format_func=FEATURES.get,
    )
    features = [f for f in FEATURES if f in chosen]  # 항상 같은 순서로 맞춰요.
    if len(features) < 2:
        st.info("속성을 두 개 이상 골라 주세요. 그래야 모델을 만들고 채점할 수 있어요.")
        st.stop()

    train, test = split_train_test(df)
    models = fit_models(train, features)
    y_test = test[TARGET].astype(int).to_numpy()
    predictions = predict_all(models, test[features])
    counts = {name: confusion_counts(y_test, pred) for name, pred in predictions.items()}
    st.caption(
        f"훈련용 {len(train)}명으로 학습하고, 테스트용 {len(test)}명으로 채점해요. "
        "(번호 순으로 10명씩 묶어 묶음마다 앞쪽을 테스트용으로 고정하는 방법은 '분류 모델' 페이지와 같아요.)"
    )

    # 2) 혼동행렬
    st.subheader("2. 혼동행렬 (테스트 데이터)")
    left, right = st.columns(2)
    for column, name in zip((left, right), (NAME_LOGIT, NAME_TREE)):
        column.markdown(f"**{name}**")
        column.dataframe(confusion_table(counts[name]), hide_index=True)
    st.caption(
        "행은 실제, 열은 예측이에요. "
        "TP는 수면 장애를 맞게 찾은 사람, FN은 수면 장애인데 아니라고 한 사람(놓침), "
        "FP는 아닌데 수면 장애라고 한 사람(헛짚음), TN은 아니라고 맞게 답한 사람이에요."
    )

    # 3) 지표 비교 표
    st.subheader("3. 네 가지 지표 비교")
    n_test = len(test)
    n_positive = int((y_test == 1).sum())
    st.markdown(f"테스트용 **{n_test}명** 중 실제 수면 장애인 사람은 **{n_positive}명**이에요.")

    rows = []
    for name in (NAME_LOGIT, NAME_TREE, NAME_BASE):
        values = metric_values(counts[name])
        rows.append(
            {
                "모델": name,
                "정확도": percent(values["정확도"]),
                "재현율": percent(values["재현율"]),
                "정밀도": percent(values["정밀도"]),
                "F1": percent(values["F1"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "정확도 = (TP + TN) ÷ 전체 사람 수 · "
        "재현율 = TP ÷ 실제 수면 장애인 사람 수 · "
        "정밀도 = TP ÷ 수면 장애라 예측한 사람 수 · "
        "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
        f"수면 장애라 예측한 사람이 0명이면 정밀도와 F1은 '{UNDEFINED}'이라고 적어요."
    )

    # 4) 놓친 환자 / 헛짚은 사람
    st.subheader("4. 틀린 사람들 살펴보기")
    chosen_name = st.radio("목록을 볼 모델", [NAME_LOGIT, NAME_TREE], horizontal=True)
    pred = predictions[chosen_name]
    missed_count, missed_table = person_list(test, (y_test == 1) & (pred == 0))
    wrong_count, wrong_table = person_list(test, (y_test == 0) & (pred == 1))

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

    c = counts[chosen_name]
    if missed_count == c["FN"] and wrong_count == c["FP"]:
        st.markdown(
            f"왼쪽 목록의 전체 인원 {missed_count}명은 혼동행렬의 FN 칸({c['FN']}명)과, "
            f"오른쪽 목록의 전체 인원 {wrong_count}명은 FP 칸({c['FP']}명)과 같아요."
        )
    else:
        st.warning(
            f"목록의 인원(왼쪽 {missed_count}명, 오른쪽 {wrong_count}명)이 "
            f"혼동행렬의 FN({c['FN']}명), FP({c['FP']}명)와 달라요."
        )


main()
