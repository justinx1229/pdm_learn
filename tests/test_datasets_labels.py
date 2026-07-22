import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm_learn.datasets import (
    DatasetSpec,
    align_matrices,
    load_matrices,
    matrix_id_column,
    prepare_matrix,
)
from pdm_learn.labels import balance_labeled_rows, load_label_set, split_labeled_rows


class DatasetUtilitiesTests(unittest.TestCase):
    def test_prepare_matrix_uses_named_identifier_and_row_centers_values(self) -> None:
        raw = pd.DataFrame(
            {
                "sample_1": [1.0, 2.0],
                "record id": [" A ", "B"],
                "sample_2": [3.0, 6.0],
            }
        )

        prepared = prepare_matrix(raw, DatasetSpec("measurements", id_column="record id"))

        self.assertEqual(prepared.columns.tolist(), ["record id", "sample_1", "sample_2"])
        self.assertEqual(
            matrix_id_column(prepared, DatasetSpec("measurements", id_column="record id")),
            "record id",
        )
        self.assertEqual(prepared["record id"].tolist(), ["A", "B"])
        self.assertEqual(
            prepared[["sample_1", "sample_2"]].values.tolist(), [[-1.0, 1.0], [-2.0, 2.0]]
        )

    def test_load_matrices_accepts_specs_for_arbitrary_identifier_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.csv"
            pd.DataFrame({"value": [0, 1], "entity": [" A ", "B"]}).to_csv(path, index=False)

            matrices = load_matrices(
                {" state ": path},
                specs=[
                    DatasetSpec(
                        "state", kind="binary", levels=(0, 1), normalize=None, id_column="entity"
                    )
                ],
            )

        self.assertEqual(set(matrices), {"state"})
        self.assertEqual(matrices["state"].columns.tolist(), ["entity", "value"])
        self.assertEqual(matrices["state"]["entity"].tolist(), ["A", "B"])

    def test_load_matrices_without_specs_does_not_normalize_unknown_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "binary.csv"
            pd.DataFrame({"entity": ["A"], "S1": [0], "S2": [1]}).to_csv(path, index=False)

            matrices = load_matrices({"event": path})

        self.assertEqual(matrices["event"].iloc[0, 1:].tolist(), [0, 1])

    def test_align_matrices_matches_ids_and_samples_without_fixed_headers(self) -> None:
        left = pd.DataFrame(
            {
                "S1": [1.0, 2.0],
                "left entity": ["B", "A"],
                "S2": [3.0, 4.0],
            }
        )
        right = pd.DataFrame(
            {
                "right entity": ["A", "B"],
                " S2 ": [30.0, 40.0],
                "S1": [10.0, 20.0],
            }
        )

        aligned_left, aligned_right = align_matrices(
            left,
            right,
            DatasetSpec("left", normalize=None, id_column="left entity"),
            DatasetSpec("right", normalize=None, id_column="right entity"),
        )

        self.assertEqual(aligned_left.columns.tolist(), ["left entity", "S1", "S2"])
        self.assertEqual(aligned_right.columns.tolist(), ["right entity", "S1", "S2"])
        self.assertEqual(aligned_left["left entity"].tolist(), ["B", "A"])
        self.assertEqual(aligned_right["right entity"].tolist(), ["B", "A"])
        self.assertEqual(aligned_right[["S1", "S2"]].values.tolist(), [[20.0, 40.0], [10.0, 30.0]])


class LabelUtilitiesTests(unittest.TestCase):
    def test_load_label_set_combines_files_and_in_memory_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.csv"
            second = Path(temp_dir) / "second.csv"
            pd.Series([" A ", "B"]).to_csv(first, index=False, header=False)
            pd.Series(["C"]).to_csv(second, index=False, header=False)

            labels = load_label_set(
                [first, second],
                labels=[" D ", ""],
            )

        self.assertEqual(labels, {"A", "B", "C", "D"})

    def test_split_and_balance_labeled_rows_support_custom_identifier_column(self) -> None:
        table = pd.DataFrame(
            {
                "feature": [1.0, 2.0, 3.0, None],
                "record_id": ["A", "B", "C", "D"],
            }
        )

        positive, negative = split_labeled_rows(table, [" A "], id_column="record_id")
        balanced_positive, balanced_negative = balance_labeled_rows(
            pd.concat([positive, positive], ignore_index=True),
            negative,
            random_state=7,
        )

        self.assertEqual(positive["record_id"].tolist(), ["A"])
        self.assertEqual(negative["record_id"].tolist(), ["B", "C"])
        self.assertEqual(len(balanced_positive), len(balanced_negative))
        self.assertEqual(balanced_negative["record_id"].tolist(), ["B", "C"])


if __name__ == "__main__":
    unittest.main()
