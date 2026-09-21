import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _prediction_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ppi" / "run_predictions.py"
    spec = spec_from_file_location("ppi_run_predictions", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PpiPredictionControlTests(unittest.TestCase):
    def test_benchmark_uses_only_explicit_controls_and_excludes_label_metadata(self) -> None:
        positive = pd.DataFrame(
            {
                "Gene1": ["A", "B", "C", "D"],
                "Gene2": ["E", "F", "G", "H"],
                "Complex Name": ["first", "first", "second", "second"],
                "Y": [1, 1, 1, 1],
                "Confidence_Score": [0.9, 0.8, 0.7, 0.6],
                "gene_exp.gene_mut.0": [0.9, 0.8, 0.7, 0.6],
                "gene_exp.gene_mut.1": [0.8, 0.7, 0.9, 0.6],
            }
        )
        negative = positive.copy()
        negative.loc[:, "Gene1"] = ["I", "J", "K", "L"]
        negative.loc[:, "Y"] = 0
        negative.loc[:, "Confidence_Score"] = [0.1, 0.2, 0.3, 0.4]
        negative.loc[:, "gene_exp.gene_mut.0"] = [0.1, 0.2, 0.3, 0.4]
        negative.loc[:, "gene_exp.gene_mut.1"] = [0.2, 0.3, 0.1, 0.4]

        result = _prediction_script().evaluate_control_sets(
            [positive],
            [negative],
            model="logistic_regression",
            n_splits=2,
            n_features=None,
            random_state=7,
        )

        self.assertEqual(result["control"].tolist(), [1])
        self.assertEqual(result["feature_columns"].tolist(), [2])
        self.assertEqual(result["positive_rows"].tolist(), [4])
        self.assertEqual(result["negative_rows"].tolist(), [4])
        self.assertEqual(result["cv_strategy"].tolist(), ["stratified_group_kfold"])

    def test_benchmark_rejects_controls_without_complex_groups(self) -> None:
        positive = pd.DataFrame(
            {
                "gene_exp.gene_mut.0": [0.9, 0.8],
                "gene_exp.gene_mut.1": [0.8, 0.7],
            }
        )
        negative = pd.DataFrame(
            {
                "gene_exp.gene_mut.0": [0.1, 0.2],
                "gene_exp.gene_mut.1": [0.2, 0.3],
            }
        )

        with self.assertRaisesRegex(ValueError, "Complex Name"):
            _prediction_script().evaluate_control_sets(
                [positive],
                [negative],
                model="logistic_regression",
                n_splits=2,
                n_features=None,
                random_state=7,
            )

    def test_overlapping_complexes_are_held_out_together(self) -> None:
        script = _prediction_script()
        positive = pd.DataFrame({
            "canonical_pair": ["A.B", "B.C", "D.E"],
            "Complex Name": ["first", "second", "third"],
            "gene_exp.gene_mut.0": [0.9, 0.8, 0.7],
        })
        negative = pd.DataFrame({
            "canonical_pair": ["X.Y", "X.Z", "Y.Z"],
            "gene_exp.gene_mut.0": [0.1, 0.2, 0.3],
        })
        # Shared catalogue pairs connect groups even when not sampled as controls.
        catalogue = {"first": {"A.B", "A.C"}, "second": {"B.C", "A.C"}, "third": {"D.E"}}
        with patch.object(
            script, "evaluate_labeled_features", return_value=pd.DataFrame([{}])
        ) as cv:
            script.evaluate_control_sets(
                positive, negative, complex_pairs=catalogue,
                model="logistic_regression", n_splits=2, n_features=None, random_state=7,
            )
        groups = cv.call_args.kwargs["groups"]
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        self.assertEqual(len(set(groups[3:])), 3)

        for invalid, message in (
            (negative.assign(canonical_pair=["A.C", "X.Z", "Y.Z"]), "known positive"),
            (negative.assign(canonical_pair=["X.Y", "X.Y", "Y.Z"]), "unique"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                script.evaluate_control_sets(
                    positive, invalid, complex_pairs=catalogue,
                    model="logistic_regression", n_splits=2, n_features=None, random_state=7,
                )

    def test_transitive_holdout_groups_do_not_depend_on_catalogue_order(self) -> None:
        script = _prediction_script()
        catalogue = {"first": {"A.B"}, "second": {"B.C"}, "bridge": {"A.B", "B.C"}}
        expected = dict.fromkeys(catalogue, "bridge")
        self.assertEqual(script._complex_holdout_groups(catalogue), expected)
        self.assertEqual(
            script._complex_holdout_groups(dict(reversed(catalogue.items()))), expected
        )
