from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


def _validate_inputs(positive: np.ndarray, negative: np.ndarray, trials: int) -> str | None:
    if trials < 2:
        return "unable to create complete graph"
    if len(positive) > len(negative):
        return "unable to partition negatives"
    if len(positive[0]) != len(negative[0]):
        return "inconsistent feature lengths"
    return None


def _normalize_model_name(model: str) -> str:
    key = model.upper().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "SVR": "SVR",
        "GBR": "GBR",
        "GRADIENTBOOSTEDREGRESSION": "GBR",
        "XGB": "XGB",
        "XGBOOST": "XGB",
        "MLP": "MLP",
        "MULTILAYEREDPERCEPTRON": "MLP",
        "LR": "LR",
        "LOGISTICREGRESSION": "LR",
        "GNB": "GNB",
        "GAUSSIANNB": "GNB",
        "GAUSSIANNAIVEBAYES": "GNB",
        "DTR": "DTR",
        "DECISIONTREEREGRESSOR": "DTR",
    }
    return aliases.get(key, model)


def _build_predictor(model: str):
    model_name = _normalize_model_name(model)
    if model_name == "SVR":
        return make_pipeline(StandardScaler(), SVR())
    if model_name == "GBR":
        return GradientBoostingRegressor(random_state=42)
    if model_name == "XGB":
        if xgb is None:
            raise ImportError("xgboost is required when model='XGB'")
        return xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
    if model_name == "MLP":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=2000,
                early_stopping=False,
                random_state=42,
            ),
        )
    if model_name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    if model_name == "GNB":
        return GaussianNB()
    if model_name == "DTR":
        return DecisionTreeRegressor(random_state=42)
    raise ValueError(f"Unsupported model: {model}")


def _predict_scores(predictor, X: np.ndarray) -> np.ndarray:
    if hasattr(predictor, "predict_proba"):
        probabilities = predictor.predict_proba(X)
        if probabilities.ndim == 2:
            classes = getattr(predictor, "classes_", None)
            if classes is not None:
                positive_matches = np.where(np.asarray(classes) == 1)[0]
                if len(positive_matches):
                    return probabilities[:, positive_matches[0]]
            return probabilities[:, -1]
        return np.asarray(probabilities, dtype=float)
    if hasattr(predictor, "decision_function"):
        return np.asarray(predictor.decision_function(X), dtype=float)
    return np.asarray(predictor.predict(X), dtype=float)


def _coerce_feature_matrix(data: Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        feature_frame = data
        if feature_frame.shape[1] > 1 and not pd.api.types.is_numeric_dtype(feature_frame.dtypes.iloc[0]):
            feature_frame = feature_frame.iloc[:, 1:]
        return feature_frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    array = np.asarray(data)
    if array.ndim == 2 and array.shape[1] > 1 and not np.issubdtype(array.dtype, np.number):
        try:
            return np.asarray(array[:, 1:], dtype=float)
        except (TypeError, ValueError):
            return pd.DataFrame(array[:, 1:]).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.asarray(array, dtype=float)


def _select_features_with_ks_test(
    positive: np.ndarray,
    negative: np.ndarray,
    features_left: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if features_left is None:
        features_left = 2 * len(positive)
    features_left = max(1, min(features_left, positive.shape[1]))

    p_val = []
    for j in range(0, len(positive[0]) - 1):
        _, p_value = stats.kstest(positive[:, j], negative[:, j])
        p_val.append(p_value)

    if not p_val or features_left >= len(p_val):
        return positive, negative

    drop_ind = np.argpartition(p_val, features_left - 1)[features_left:]
    return np.delete(positive, drop_ind, axis=1), np.delete(negative, drop_ind, axis=1)


def core_predict(
    positive: Sequence[Sequence[float]],
    negative: Sequence[Sequence[float]],
    trials: int,
    model: str = "GBR",
    ks_test: bool = False,
    features_left: int | None = None,
):
    positive_array = np.asarray(positive)
    negative_array = np.asarray(negative)

    error = _validate_inputs(positive_array, negative_array, trials)
    if error is not None:
        return error

    pos = np.copy(positive_array)
    neg = np.copy(negative_array)

    index = 0
    size = len(pos)
    scores = np.zeros(len(neg))
    number = np.zeros(len(neg))

    if ks_test:
        pos, neg = _select_features_with_ks_test(pos, neg, features_left)

    pos = np.concatenate([pos, np.ones(len(positive_array))[:, np.newaxis]], axis=1)
    neg = np.concatenate(
        [
            np.arange(len(negative_array))[:, np.newaxis],
            neg,
            np.zeros(len(negative_array))[:, np.newaxis],
        ],
        axis=1,
    )

    for _ in range(trials):
        mat = np.concatenate([pos, neg[index : index + size, 1:]])
        X = mat[:, :-1]
        y = mat[:, -1]

        predictor = _build_predictor(model).fit(X, y)
        neg_test = np.delete(neg, range(index, index + size), axis=0)
        for row_index, value in enumerate(_predict_scores(predictor, neg_test[:, 1:-1])):
            scores[int(neg_test[row_index, 0])] += value
            number[int(neg_test[row_index, 0])] += 1

        index += size
        if index + size > len(neg):
            index = 0
            np.random.shuffle(neg)

    return scores / number


def LOOCV(
    positive: Sequence[Sequence[float]],
    negative: Sequence[Sequence[float]],
    trials: int,
    model: str = "GBR",
    ks_test: bool = False,
    features_left: int | None = None,
    graph: bool = False,
    equation: bool = False,
):
    positive_array = np.asarray(positive)
    negative_array = np.asarray(negative)

    ranks = []
    for i in range(len(positive_array)):
        result = np.concatenate(
            [
                [np.append(np.zeros(len(negative_array)), 1)],
                [
                    core_predict(
                        np.delete(positive_array, i, axis=0),
                        np.concatenate([negative_array, [positive_array[i]]], axis=0),
                        trials,
                        model=model,
                        ks_test=ks_test,
                        features_left=features_left,
                    )
                ],
            ],
            axis=0,
        )
        result = result[:, np.flip(result[1].argsort())]
        ranks.append(np.where(result[0] == 1)[0][0])

    values, base = np.histogram(ranks, bins=len(negative_array) + 1, range=(0, len(negative_array) + 1))
    cumulative = np.cumsum(values)
    area = np.sum(cumulative) / (len(positive_array) * (len(negative_array) + 1))

    if graph:
        plt.title("area = " + str(area))
        plt.step(base[:-1], cumulative, where="post")
        plt.show()

    if equation:
        return area, base[:-1], cumulative
    return area


def LOOCV_grouped_plot(
    data_dict: Mapping[str, tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]],
    trials: int,
    models: Sequence[str] = ("SVR", "XGB", "GBR", "MLP", "LR", "GNB"),
    ks_test: bool = False,
    features_left: int | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> np.ndarray:
    methods = list(data_dict.keys())
    results = np.zeros((len(methods), len(models)))
    results[:] = np.nan

    for i, method in enumerate(methods):
        positive, negative = data_dict[method]
        positive_values = _coerce_feature_matrix(positive)
        negative_values = _coerce_feature_matrix(negative)

        if (
            positive_values.ndim != 2
            or negative_values.ndim != 2
            or len(positive_values) == 0
            or len(negative_values) == 0
            or positive_values.shape[1] == 0
            or negative_values.shape[1] == 0
        ):
            print(f"Skipping {method}: insufficient usable feature data")
            continue

        for j, model in enumerate(models):
            print(f"Running {method} - {model}")
            try:
                results[i, j] = LOOCV(
                    positive_values,
                    negative_values,
                    trials,
                    model=model,
                    ks_test=ks_test,
                    features_left=features_left,
                    graph=False,
                )
            except Exception as exc:
                print(f"Skipping {method} - {model}: {exc}")

    x = np.arange(len(methods))
    width = 0.8 / max(len(models), 1)
    plt.figure(figsize=figsize)

    for j, model in enumerate(models):
        plt.bar(x + j * width, results[:, j], width, label=model)

    plt.xticks(x + width * (len(models) - 1) / 2, methods)
    plt.ylabel("LOOCV AUC")
    plt.title("LOOCV AUC by Data Conversion Method and Model")
    plt.legend()
    plt.ylim(0.5, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return results


def KFold_PR(
    positive: Sequence[Sequence[float]],
    negative: Sequence[Sequence[float]],
    trials: int,
    model: str = "GBR",
    ks_test: bool = False,
    features_left: int | None = None,
    n_splits: int = 5,
    graph: bool = False,
    random_state: int = 42,
):
    positive_array = np.asarray(positive)
    negative_array = np.asarray(negative)

    X = np.vstack([positive_array, negative_array])
    y = np.concatenate([np.ones(len(positive_array)), np.zeros(len(negative_array))])

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_scores = []
    all_true = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        pos_train = X_train[y_train == 1]
        neg_train = X_train[y_train == 0]
        pos_test = X_test[y_test == 1]
        neg_test = X_test[y_test == 0]

        combined_test = np.vstack([neg_test, pos_test])
        scores = core_predict(
            pos_train,
            np.vstack([neg_train, combined_test]),
            trials,
            model=model,
            ks_test=ks_test,
            features_left=features_left,
        )

        test_scores = scores[len(neg_train) :]
        test_labels = np.concatenate([np.zeros(len(neg_test)), np.ones(len(pos_test))])
        all_scores.extend(test_scores)
        all_true.extend(test_labels)

    all_scores = np.array(all_scores)
    all_true = np.array(all_true)

    precision, recall, _ = precision_recall_curve(all_true, all_scores)
    average_precision = average_precision_score(all_true, all_scores)

    if graph:
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve (AP = {average_precision:.4f})")
        plt.show()

    return average_precision, recall, precision


def area_table(
    positive: Sequence[Sequence[float]],
    negative: Sequence[Sequence[float]],
    trials: int,
    model: str = "SVR",
    feat_arr: Sequence[int] = (),
) -> list[float]:
    areas = []
    for value in feat_arr:
        print(value)
        areas.append(LOOCV(positive, negative, trials, model=model, ks_test=True, features_left=value))
    return areas


def ks_pvalue(pos: Sequence[Sequence[float]], neg: Sequence[Sequence[float]]):
    pos_array = np.asarray(pos)
    neg_array = np.asarray(neg)

    if len(pos_array[0]) != len(neg_array[0]):
        return "inconsistent feature lengths"

    p_val = np.zeros(len(pos_array[0]))
    for j in range(0, len(pos_array[0])):
        _, p_value = stats.kstest(pos_array[:, j], neg_array[:, j])
        p_val[j] = p_value
    return p_val


def heatmap(
    data: Sequence[float],
    dimensions: Sequence[Sequence[int]],
    cmap: str = "hot",
    min: float | None = None,
    max: float | None = None,
    flip: bool = False,
    axes: bool = False,
    colorbar: bool = False,
) -> None:
    index = 0
    for dim in dimensions:
        matrix = np.reshape(data[index : index + dim[0] * dim[1]], (dim[0], dim[1]))
        if flip:
            matrix = np.flip(matrix, 0)
        if min is not None and max is not None:
            image = plt.imshow(matrix, cmap=cmap, interpolation="nearest", vmin=min, vmax=max)
        else:
            image = plt.imshow(matrix, cmap=cmap, interpolation="nearest")
        if colorbar:
            plt.colorbar(image)
        if not axes:
            plt.axis("off")
        plt.show()
        index += dim[0] * dim[1]


def importance_test(
    positive: Sequence[Sequence[float]],
    negative: Sequence[Sequence[float]],
    trials: int,
    isks: bool,
    num_left: int | None = None,
) -> list[np.ndarray] | str:
    positive_array = np.asarray(positive)
    negative_array = np.asarray(negative)

    error = _validate_inputs(positive_array, negative_array, trials)
    if error is not None:
        return error

    pos = np.concatenate([np.copy(positive_array), np.ones(len(positive_array))[:, np.newaxis]], axis=1)
    neg = np.concatenate(
        [
            np.arange(len(negative_array))[:, np.newaxis],
            np.copy(negative_array),
            np.zeros(len(negative_array))[:, np.newaxis],
        ],
        axis=1,
    )

    index = 0
    size = len(pos)
    out = []

    for trial_index in range(trials):
        mat = np.concatenate([pos, neg[index : index + size, 1:]])
        features = mat[:, :-1]
        labels = mat[:, -1]

        if isks:
            feature_values = features[:, :-1]
            keep = num_left if num_left is not None else feature_values.shape[1]
            filtered_pos, filtered_neg = _select_features_with_ks_test(
                feature_values[:size],
                feature_values[size:],
                keep,
            )
            features = np.vstack(
                [
                    np.concatenate([filtered_pos, features[:size, -1][:, np.newaxis]], axis=1),
                    np.concatenate([filtered_neg, features[size:, -1][:, np.newaxis]], axis=1),
                ]
            )

        predictor = SVR().fit(features, labels)
        results = permutation_importance(
            predictor,
            features,
            labels,
            n_repeats=10,
            random_state=trial_index,
        )
        importance = results.importances_mean
        plt.bar([value for value in range(len(importance))], importance)
        plt.show()
        out.append(importance)

        index += size
        if index + size > len(neg):
            index = 0
            np.random.shuffle(neg)

    return out
