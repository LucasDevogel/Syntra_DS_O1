"""Checks on the real RT file: catch surprises in the data that hand-made test tables cannot show.

Skipped when the raw file is not present.
"""
import pytest

from pipeline import MIN_AUDIENCE_RATINGS, MIN_CRITIC_REVIEWS, RAW_PATH, load_raw, run_pipeline

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason=f"raw data not found at {RAW_PATH}")


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def films(raw):
    return run_pipeline(raw)[0]


@pytest.mark.parametrize("column", ["tomatometer_rating", "audience_rating"])
def test_raw_scores_are_whole_numbers_between_0_and_100(raw, column):
    scores = raw[column].dropna()
    assert scores.between(0, 100).all()
    assert (scores % 1 == 0).all()


@pytest.mark.parametrize("column", ["critic_score", "audience_score"])
def test_pipeline_scores_are_present_and_between_0_and_100(films, column):
    assert films[column].notna().all()
    assert films[column].between(0, 100).all()


def test_pipeline_review_counts_meet_thresholds(films):
    assert (films["critic_count"] >= MIN_CRITIC_REVIEWS).all()
    assert (films["audience_count"] >= MIN_AUDIENCE_RATINGS).all()


def test_pipeline_film_ids_are_unique(films):
    assert films["film_id"].is_unique
