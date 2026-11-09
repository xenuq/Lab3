"""Воспроизводит все числа отчёта: python run_experiments.py"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # только для сверки, не внутри собственной модели
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    classification_metrics,
    compare_classical_models,
    sklearn_logistic_metrics,
    train_linear_regression,
    train_logistic_classifier,
)

SEED = 42
TEST_SIZE = 0.2
log: dict = {}

# ---------------- 1. Линейная регрессия ----------------
st = pd.read_csv("Student_Performance.csv")
log["student_rows"] = len(st)
log["student_duplicates"] = int(st.duplicated().sum())
log["student_missing"] = int(st.isna().sum().sum())
st["Extracurricular Activities"] = st["Extracurricular Activities"].map({"No": 0, "Yes": 1})
X = st.drop(columns="Performance Index")
y = st["Performance Index"]
feature_names = list(X.columns)
Xtr, Xte, ytr, yte = train_test_split(X.values, y.values, test_size=TEST_SIZE, random_state=SEED)
log["reg_split"] = {"train": len(ytr), "test": len(yte)}
log["reg_target_mean_std"] = [float(y.mean()), float(y.std())]
log["reg_mse_of_mean_predictor"] = float(np.mean((yte - ytr.mean()) ** 2))


def run_reg(lr, epochs):
    r = train_linear_regression(RegressionInput(Xtr, ytr, Xte, yte, lr, epochs))
    return r, float(mean_squared_error(yte, r.predictions))


reg_grid = {}
for lr, ep in [(0.001, 800), (0.01, 800), (0.05, 800), (0.2, 800), (0.05, 100), (0.05, 3000)]:
    r, mse = run_reg(lr, ep)
    reg_grid[f"lr={lr},epochs={ep}"] = {"train_mse_final": r.loss_history[-1], "test_mse": mse}
log["reg_grid"] = reg_grid

res, test_mse = run_reg(0.05, 800)
h = res.loss_history
log["reg_final"] = {
    "test_mse": test_mse,
    "test_r2": float(r2_score(yte, res.predictions)),
    "mse_epoch_1": h[0], "mse_epoch_10": h[9], "mse_epoch_50": h[49],
    "mse_epoch_100": h[99], "mse_epoch_800": h[-1],
    "bias": res.bias, "weights": dict(zip(feature_names, map(float, res.weights))),
}
sk = LinearRegression().fit(Xtr, ytr)
log["reg_reference_sklearn_closed_form_test_mse"] = float(mean_squared_error(yte, sk.predict(Xte)))

# без масштабирования внутри функции нельзя показать напрямую, поэтому сравниваем
# с «сырыми» признаками, подав их в тот же алгоритм через ручной GD без стандартизации
def raw_gd(lr, epochs):
    n = len(ytr); w = np.zeros(Xtr.shape[1]); b = 0.0
    with np.errstate(all="ignore"):
        for _ in range(epochs):
            e = Xtr @ w + b - ytr
            w = w - lr * 2 / n * Xtr.T @ e
            b = b - lr * 2 / n * e.sum()
        mse = float(np.mean((Xte @ w + b - yte) ** 2))
        return mse if np.isfinite(mse) else "diverged (inf/NaN)"
log["reg_no_scaling_test_mse_lr0.05"] = raw_gd(0.05, 800)
log["reg_no_scaling_test_mse_lr0.001"] = raw_gd(0.001, 800)

# ---------------- 2. Логистическая регрессия ----------------
g = pd.read_csv("german.csv", sep=";")
yg = g["Creditability"].values
Xg = g.drop(columns="Creditability")
nominal = ["Purpose", "Sex_Marital_Status", "Guarantors", "Most_valuable_available_asset",
           "Concurrent_Credits", "Type_of_apartment", "Occupation"]
Xg_ord = Xg.copy().astype(float)
Xg_oh = pd.get_dummies(Xg, columns=nominal, dtype=float)
log["german"] = {"rows": len(g), "class_balance": {int(k): int(v) for k, v in pd.Series(yg).value_counts().items()},
                 "features_ordinal_variant": Xg_ord.shape[1], "features_onehot_variant": Xg_oh.shape[1]}


def split_g(Xdf):
    return train_test_split(Xdf.values, yg, test_size=TEST_SIZE, random_state=SEED, stratify=yg)


def run_clf(Xdf, lr, ep):
    a, b_, c, d = split_g(Xdf)
    r = train_logistic_classifier(ClassificationInput(a, c, b_, d, lr, ep))
    return r, classification_metrics(d, r.probabilities, r.predictions)


clf_grid = {}
for name, Xdf in [("ordinal_codes", Xg_ord), ("one_hot_nominal", Xg_oh)]:
    for lr, ep in [(0.1, 1000), (0.01, 1000), (0.5, 1000), (0.1, 100)]:
        r, m = run_clf(Xdf, lr, ep)
        clf_grid[f"{name},lr={lr},epochs={ep}"] = {**m.__dict__, "final_train_logloss": r.loss_history[-1]}
log["clf_grid"] = clf_grid

# финальная конфигурация выбрана по train-CV-логике: one-hot + параметры по умолчанию контракта
a, b_, c, d = split_g(Xg_oh)
inp = ClassificationInput(a, c, b_, d)  # lr=0.1, epochs=1000
own, own_m = run_clf(Xg_oh, 0.1, 1000)
lh = own.loss_history
log["clf_final"] = {"logloss_epoch_1": lh[0], "logloss_epoch_10": lh[9], "logloss_epoch_100": lh[99],
                    "logloss_epoch_1000": lh[-1], "metrics": own_m.__dict__,
                    "test_positive_rate": float(d.mean()), "pred_positive_rate": float(own.predictions.mean())}
sk_m = sklearn_logistic_metrics(inp)
cmp = compare_classical_models(inp)

results = {
    "linear_regression": {"test_mse": test_mse},
    "classification": {
        "own_logistic": cmp.logistic_regression.__dict__,
        "logistic_regression": sk_m.__dict__,
        "decision_tree": cmp.decision_tree.__dict__,
        "knn": cmp.knn.__dict__,
    },
}
assert np.isclose(results["classification"]["own_logistic"]["roc_auc"], own_m.roc_auc)
json.dump(results, open("experiment_results.json", "w"), ensure_ascii=False, indent=2)
json.dump(log, open("experiments_log.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(results, indent=2))
print(json.dumps(log, indent=2, ensure_ascii=False))
