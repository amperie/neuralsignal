import pandas as pd
import pytest

from neuralsignal.training import select_feature_columns, train_s1


def test_select_feature_columns_filters_by_feature_set():
    columns = ["example_id", "zones__a", "zones__b", "logit-lens__x", "label"]

    selected = select_feature_columns(columns, include_sets=["zones"])

    assert selected == ["zones__a", "zones__b"]


def test_select_feature_columns_supports_explicit_include_and_exclude():
    columns = ["zones__a", "zones__b", "layer_distribution__x", "label"]

    selected = select_feature_columns(
        columns,
        include_sets=["zones"],
        include_columns=["layer_distribution__x"],
        exclude_columns=["zones__b"],
    )

    assert selected == ["zones__a", "layer_distribution__x"]


def test_train_s1_from_feature_directory(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    pd.DataFrame({
        "zones__a": [0, 0.1, 0.2, 0.3, 3.0, 3.1, 3.2, 3.3],
        "layer_distribution__x": [1, 1, 1, 1, 2, 2, 2, 2],
        "label": [0, 0, 0, 0, 1, 1, 1, 1],
    }).to_parquet(features_dir / "part-00000.parquet", index=False)

    result = train_s1(tmp_path, label_column="label", feature_config={"include_sets": ["zones"]}, mlflow_config={"enabled": False}, output_root=tmp_path / "s1")

    assert result.feature_columns == ["zones__a"]
    assert result.metrics["feature_count"] == 1
    assert result.metrics["train_rows"] == 6
    assert result.metrics["test_rows"] == 2


def test_train_s1_fails_when_no_features_selected(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"label": [0, 1, 0, 1]}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="No feature columns selected"):
        train_s1(path, label_column="label")

