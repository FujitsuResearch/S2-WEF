# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""Preprocessing of the UCI Adult census table.

This is an implementation of the preprocessing pipeline that is
commonly applied to the Adult dataset in the fairness and federated-learning
literature, and which is described publicly at
https://www.valentinmihov.com/2015/04/17/adult-income-data-set/

"""

import os
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


#: Column names of the raw UCI files, which ship without a header row.
COLUMN_NAMES: List[str] = [
    "Age", "Workclass", "fnlwgt", "Education", "Education-Num",
    "Martial Status", "Occupation", "Relationship", "Race", "Sex",
    "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target",
]

#: Name of the label column.
LABEL_COLUMN = "Target"

#: Continuous columns, standardized rather than one-hot encoded.
NUMERIC_COLUMNS: List[str] = [
    "Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week",
]

#: Columns removed before encoding. ``Education`` is redundant with
#: ``Education-Num``; ``fnlwgt`` is a sampling weight, not a feature.
REDUNDANT_COLUMNS: List[str] = ["Education", "fnlwgt"]

#: Income strings and the label they map to. The test file appends a period.
INCOME_LABELS = {"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1}

#: Dtype of the one-hot columns. Pinned rather than left to the pandas
#: default, which changed from uint8 to bool in pandas 2.0. A boolean column
#: survives the CSV round trip as bool, which would make the feature matrix
#: object-typed once the numeric columns are standardized.
DUMMY_DTYPE = "uint8"

#: Seed of the row shuffle. Fixed so that the split is stable across runs.
SHUFFLE_SEED = 1234

#: Fraction of rows assigned to the training part.
TRAIN_FRACTION = 0.8

#: Separator pattern of the raw files: a comma with optional surrounding
#: whitespace. Requires the Python parser engine.
RAW_SEPARATOR = r"\s*,\s*"

DEFAULT_CACHE_PATH = "./data/adult_datasets/adult.csv"
DEFAULT_TRAIN_PATH = "./data/adult_datasets/adult.data"
DEFAULT_TEST_PATH = "./data/adult_datasets/adult.test"

UCI_TRAIN_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
UCI_TEST_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)


def load_adult(
    cache_path: str = DEFAULT_CACHE_PATH,
    train_path: str = DEFAULT_TRAIN_PATH,
    test_path: str = DEFAULT_TEST_PATH,
    train_fraction: float = TRAIN_FRACTION,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return the Adult table as standardized train and test parts.

    The encoded table is cached at ``cache_path`` on first use. When the
    cache is absent, the raw files are read from ``train_path`` and
    ``test_path``, either path falling back to its UCI URL if the local file
    is missing.

    Returns
    -------
    tuple
        ``(train_features, train_labels, test_features, test_labels)``.
        Features are one-hot encoded with the five numeric columns
        standardized; labels are floats in ``{0.0, 1.0}``.
    """
    if not os.path.isfile(cache_path):
        encoded = _encode_raw_files(train_path, test_path)
        _write_cache(encoded, cache_path)

    table = pd.read_csv(cache_path)
    table = _balance_classes(table)
    table = _shuffle(table)

    labels = table[LABEL_COLUMN].astype("float")
    features = table.drop(columns=[LABEL_COLUMN])

    return _split_and_standardize(features, labels, train_fraction)


# ----------------------------------------------------------------------
# Building the encoded table from the raw files
# ----------------------------------------------------------------------
def _encode_raw_files(train_path: str, test_path: str) -> pd.DataFrame:
    """Read the raw UCI files and return the one-hot encoded table.

    Raises
    ------
    ValueError
        If the income column holds a value outside :data:`INCOME_LABELS`,
        which would otherwise become a silent missing label.
    """
    train_source = train_path if os.path.isfile(train_path) else UCI_TRAIN_URL
    test_source = test_path if os.path.isfile(test_path) else UCI_TEST_URL
    print(f"[adult] reading raw data from {train_source} and {test_source}")

    train_rows = pd.read_csv(
        train_source, names=COLUMN_NAMES, sep=RAW_SEPARATOR,
        engine="python", na_values="?",
    )
    test_rows = pd.read_csv(
        test_source, names=COLUMN_NAMES, sep=RAW_SEPARATOR,
        engine="python", na_values="?", skiprows=1,
    )

    table = pd.concat([train_rows, test_rows]).dropna()

    labels = table[LABEL_COLUMN].map(INCOME_LABELS)
    if labels.isna().any():
        unknown = sorted(table.loc[labels.isna(), LABEL_COLUMN].unique())
        raise ValueError(
            f"unrecognized income values in the raw data: {unknown}; "
            f"expected one of {sorted(INCOME_LABELS)}"
        )
    table[LABEL_COLUMN] = labels.astype("int64")

    table = _balance_classes(table)
    table = _shuffle(table)
    table = table.drop(columns=REDUNDANT_COLUMNS)

    categorical = [
        column for column in table.columns
        if column not in NUMERIC_COLUMNS and column != LABEL_COLUMN
    ]
    return pd.get_dummies(data=table, columns=categorical, dtype=DUMMY_DTYPE)


def _write_cache(table: pd.DataFrame, cache_path: str) -> None:
    """Save the encoded table so that later runs skip the raw files."""
    directory = os.path.dirname(cache_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    table.to_csv(cache_path, index=False)
    print(f"[adult] cached encoded table at {cache_path} "
          f"({len(table)} rows, {table.shape[1]} columns)")


# ----------------------------------------------------------------------
# Shared steps
# ----------------------------------------------------------------------
def _balance_classes(table: pd.DataFrame) -> pd.DataFrame:
    """Keep every positive row and an equal number of leading negatives.

    Raises
    ------
    ValueError
        If the table holds fewer negatives than positives, in which case no
        balanced subset of this form exists.
    """
    positives = table[table[LABEL_COLUMN] == 1]
    negatives = table[table[LABEL_COLUMN] == 0]

    if len(negatives) < len(positives):
        raise ValueError(
            f"cannot balance the table: {len(positives)} positive rows but "
            f"only {len(negatives)} negative rows"
        )

    return pd.concat([positives, negatives[: len(positives)]])


def _shuffle(table: pd.DataFrame) -> pd.DataFrame:
    """Permute the rows with the fixed seed and renumber the index."""
    return table.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)


def _split_and_standardize(
    features: pd.DataFrame,
    labels: pd.Series,
    train_fraction: float,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Cut the table in two and standardize the numeric columns.

    The scaler sees only the training rows, so no statistic of the test part
    leaks into the training features.
    """
    cut = int(train_fraction * len(features))

    train_features = features[:cut].copy(deep=True)
    test_features = features[cut:].copy(deep=True)

    scaler = StandardScaler()
    train_features[NUMERIC_COLUMNS] = scaler.fit_transform(
        train_features[NUMERIC_COLUMNS]
    )
    test_features[NUMERIC_COLUMNS] = scaler.transform(
        test_features[NUMERIC_COLUMNS]
    )

    return train_features, labels[:cut], test_features, labels[cut:]


# ----------------------------------------------------------------------
# Backwards-compatible alias
# ----------------------------------------------------------------------
def get_train_test(
    dataset_dir: str = DEFAULT_CACHE_PATH,
    train_dir: str = DEFAULT_TRAIN_PATH,
    test_dir: str = DEFAULT_TEST_PATH,
    train_test_ratio: float = TRAIN_FRACTION,
):
    """Wrapper kept for older call sites."""
    return load_adult(dataset_dir, train_dir, test_dir, train_test_ratio)