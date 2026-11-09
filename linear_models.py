import numpy as np
from sklearn.linear_model import LogisticRegression as _SkLogReg  # только для сравнения, не внутри собственных моделей
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

RANDOM_STATE = 42


def _as_2d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(len(x), -1)


def _standardizer(x_train: np.ndarray):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # константные признаки не делим на 0
    return mean, std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # устойчивая реализация: exp считается только от неположительных значений
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    x_tr = _as_2d(data.train_features)
    y_tr = np.asarray(data.train_target, dtype=float).ravel()
    x_te = _as_2d(data.test_features)

    mean, std = _standardizer(x_tr)
    xs = (x_tr - mean) / std
    n = len(y_tr)

    w = np.zeros(xs.shape[1])
    b = 0.0
    history: list[float] = []
    for _ in range(int(data.epochs)):
        err = xs @ w + b - y_tr
        history.append(float(np.mean(err ** 2)))
        w -= data.learning_rate * (2.0 / n) * (xs.T @ err)
        b -= data.learning_rate * (2.0 / n) * err.sum()

    # возвращаем параметры в исходном масштабе признаков: y = X @ w + b
    w_orig = w / std
    b_orig = float(b - np.sum(w * mean / std))
    preds = x_te @ w_orig + b_orig
    return RegressionResult(weights=w_orig, bias=b_orig, loss_history=history, predictions=preds)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    x_tr = _as_2d(data.train_features)
    y_tr = np.asarray(data.train_target, dtype=float).ravel()
    x_te = _as_2d(data.test_features)

    mean, std = _standardizer(x_tr)
    xs = (x_tr - mean) / std
    n = len(y_tr)

    w = np.zeros(xs.shape[1])
    b = 0.0
    eps = 1e-12
    history: list[float] = []
    for _ in range(int(data.epochs)):
        p = _sigmoid(xs @ w + b)
        loss = -np.mean(y_tr * np.log(p + eps) + (1.0 - y_tr) * np.log(1.0 - p + eps))
        history.append(float(loss))
        err = p - y_tr
        w -= data.learning_rate * (xs.T @ err) / n
        b -= data.learning_rate * err.sum() / n

    w_orig = w / std
    b_orig = float(b - np.sum(w * mean / std))
    proba = _sigmoid(x_te @ w_orig + b_orig)
    preds = (proba >= 0.5).astype(int)
    return ClassificationResult(
        weights=w_orig, bias=b_orig, loss_history=history, probabilities=proba, predictions=preds
    )


def classification_metrics(y_true, proba, pred) -> ClassificationMetrics:
    y_true = np.asarray(y_true).ravel()
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
    )


def sklearn_logistic_metrics(data: ClassificationInput) -> ClassificationMetrics:
    """Готовая LogisticRegression из sklearn (только как эталон для сравнения)."""
    x_tr, x_te = _as_2d(data.train_features), _as_2d(data.test_features)
    y_tr = np.asarray(data.train_target).ravel()
    mean, std = _standardizer(x_tr)
    model = _SkLogReg(max_iter=2000).fit((x_tr - mean) / std, y_tr)
    proba = model.predict_proba((x_te - mean) / std)[:, 1]
    return classification_metrics(data.test_target, proba, (proba >= 0.5).astype(int))


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    x_tr, x_te = _as_2d(data.train_features), _as_2d(data.test_features)
    y_tr = np.asarray(data.train_target).ravel().astype(int)
    y_te = np.asarray(data.test_target).ravel().astype(int)

    # 1) собственная логистическая регрессия
    own = train_logistic_classifier(data)
    own_m = classification_metrics(y_te, own.probabilities, own.predictions)

    # гиперпараметры baseline-моделей подбираются только по train (CV), test не используется
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    tree_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [3, 4, 5, 7, None], "min_samples_leaf": [1, 5, 10]},
        scoring="roc_auc", cv=cv,
    ).fit(x_tr, y_tr)
    tree = tree_search.best_estimator_
    tree_proba = tree.predict_proba(x_te)[:, 1]
    tree_m = classification_metrics(y_te, tree_proba, tree.predict(x_te))

    mean, std = _standardizer(x_tr)  # KNN чувствителен к масштабу
    xs_tr, xs_te = (x_tr - mean) / std, (x_te - mean) / std
    knn_search = GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors": [5, 11, 21, 31], "weights": ["uniform", "distance"]},
        scoring="roc_auc", cv=cv,
    ).fit(xs_tr, y_tr)
    knn = knn_search.best_estimator_
    knn_proba = knn.predict_proba(xs_te)[:, 1]
    knn_m = classification_metrics(y_te, knn_proba, knn.predict(xs_te))

    # В BaselineComparisonResult.logistic_regression кладём собственную модель (по README:
    # «сравните вашу логистическую регрессию, DecisionTree и KNN»).
    return BaselineComparisonResult(logistic_regression=own_m, decision_tree=tree_m, knn=knn_m)
