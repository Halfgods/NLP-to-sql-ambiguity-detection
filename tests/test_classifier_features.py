"""Unit tests for the ambiguity classifier representation contract.

These import the peer's classifier module directly (unlike test_ambiguity.py,
which stubs it) because this file tests the *contract* of the artifact:
question-first input text, the schema-context feature vector, and the
block-threshold loading path. No model is loaded (SentenceTransformer is
heavy); tensor-level behavior is covered by ambiguity/self_eval.py instead.
"""

import numpy as np
import pytest

from ambiguity.classifier import (
    CONTEXT_FEATURE_VERSION,
    AmbiguityClassifier,
    make_input_text,
    schema_context_features,
)

DDL_ONE_METRIC = (
    "CREATE TABLE employees (\n"
    "  employee_id INTEGER PRIMARY KEY,\n"
    "  employee_name TEXT NOT NULL,\n"
    "  salary REAL\n"
    ");"
)
DDL_THREE_RIVALS = (
    "CREATE TABLE employees (\n"
    "  employee_id INTEGER PRIMARY KEY,\n"
    "  employee_name TEXT NOT NULL,\n"
    "  salary REAL NOT NULL,\n"
    "  bonus REAL,\n"
    "  commission REAL\n"
    ");"
)
COMPACT_ONE_METRIC = (
    "table employees: employee_id INTEGER PRIMARY KEY, "
    "employee_name TEXT NOT NULL, salary REAL\n"
    "  samples: salary: 52000.0, 61000.5"
)


def test_input_text_is_question_first():
    text = make_input_text("who earns the most", DDL_ONE_METRIC)
    assert text.index("who earns the most") < text.index("CREATE TABLE")
    assert text == ("User question:\nwho earns the most\n\n"
                    "Database schema:\n" + DDL_ONE_METRIC)


def test_context_feature_version_stable():
    # Features are written into meta.json and must keep their meaning between
    # training and serving; changing the vector requires bumping the version.
    assert CONTEXT_FEATURE_VERSION == "v1_numeric_rivals"


def test_features_zero_without_question_metric_overlap():
    # "best folks" names no metric -> no rivals even with a wide schema.
    vec = schema_context_features("show me the best folks around",
                                  DDL_THREE_RIVALS)
    assert vec.shape == (5,)
    np.testing.assert_allclose(vec[:3], [0.4, 0.0, 0.0])
    # numeric set = the 3 metric cols + employee_id (an INTEGER column is
    # numeric in the raw sense; only *rival* matching cares about semantics)
    # total idents: employee_id, salary, bonus, commission
    np.testing.assert_allclose(vec[3], 4 / 60, rtol=1e-5)


def test_features_detect_named_metric_rival():
    # "salary" matches a numeric column: named-metric flag on, one rival.
    vec = schema_context_features("who earns the highest salary",
                                  DDL_THREE_RIVALS)
    assert vec[2] == 1.0
    assert vec[1] == pytest.approx(1 / 5)

    vec_one = schema_context_features("who earns the highest salary",
                                      DDL_ONE_METRIC)
    assert vec_one[2] == 1.0
    assert vec_one[1] == pytest.approx(1 / 5)


def test_features_work_on_runtime_compact_format():
    # render_compact (what analyze_ambiguity feeds at runtime) must also
    # yield a usable vector — it is the classifier's real serving format.
    vec = schema_context_features("who earns the highest salary",
                                  COMPACT_ONE_METRIC)
    assert vec[2] == 1.0
    assert vec[1] == pytest.approx(1 / 5)
    assert vec[0] == pytest.approx(0.2)   # employee_id + salary (/10)
    assert vec[3] == pytest.approx(2 / 60)   # declared identifiers: the 2 above


def test_feature_vector_is_bounded():
    huge = ("CREATE TABLE t (\n" +
            "\n".join(f"  giant_col_{i} REAL" for i in range(120)) +
            "\n);")
    vec = schema_context_features(
        "giant col sums " + " ".join(f"giant_col_{i}" for i in range(120)),
        huge)
    assert np.all(vec >= 0) and np.all(vec <= 1.5)


def test_block_thresholds_normalization_contract(tmp_path):
    # The load-path contract exercised end-to-end by self_eval; here we pin
    # the constructor normalizations a caller can rely on.
    class _StubClf:
        def predict_proba(self, x):
            return [[0.2, 0.3, 0.5]] * len(x)

        classes_ = np.array(["CLEAR", "AMBIGUOUS", "UNDERSPECIFIED"])

    clf = AmbiguityClassifier(_StubClf(), _StubClf(),
                              {"CLEAR": 0.1},
                              block_thresholds={"AMBIGUOUS": "0.62",
                                                "UNDERSPECIFIED": 0.77})
    assert clf.block_thresholds == {"AMBIGUOUS": 0.62, "UNDERSPECIFIED": 0.77}
    assert isinstance(clf.block_thresholds["AMBIGUOUS"], float)
    assert clf.thresholds == {"CLEAR": 0.1}

    legacy = AmbiguityClassifier(_StubClf(), _StubClf(), {})
    assert legacy.block_thresholds == {}
