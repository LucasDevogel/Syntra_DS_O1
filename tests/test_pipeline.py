"""Tests for the pure functions in pipeline.py, on small hand-made tables with known answers."""
import math

import pandas as pd
import pytest

from pipeline import (
    drop_low_review_counts,
    drop_missing_scores,
    drop_missing_year,
    prepare,
    run_pipeline,
    split_names,
    to_long,
)


def make_raw(**overrides):
    """One film in the raw RT column layout; keyword arguments override single fields."""
    film = {
        "rotten_tomatoes_link": "m/test_film",
        "movie_title": "Test Film",
        "original_release_date": "2005-06-15",
        "tomatometer_rating": 70.0,
        "tomatometer_count": 25.0,
        "audience_rating": 60.0,
        "audience_count": 1000.0,
        "content_rating": "PG-13",
        "runtime": 100.0,
        "genres": "Comedy, Drama",
        "directors": "Jane Director",
        "actors": "A One, B Two, C Three",
    }
    film.update(overrides)
    return film


# --- split_names ----------------------------------------------------------------------------------

def test_split_names_normal_list():
    assert split_names("Tom Hanks, Robin Wright, Gary Sinise") == ["Tom Hanks", "Robin Wright", "Gary Sinise"]


def test_split_names_keeps_suffix_attached():
    assert split_names("Sammy Davis, Jr., Frank Sinatra") == ["Sammy Davis, Jr.", "Frank Sinatra"]


@pytest.mark.parametrize("suffix", ["Jr.", "Sr.", "II", "III", "IV"])
def test_split_names_known_suffixes(suffix):
    assert split_names(f"John Smith, {suffix}, Mary Jones") == [f"John Smith, {suffix}", "Mary Jones"]


def test_split_names_removes_duplicates_keeping_first_position():
    # Real case: "Metro" lists Kim Miyori twice among its first five actors.
    assert split_names("A, B, A, C") == ["A", "B", "C"]


@pytest.mark.parametrize("empty", [float("nan"), None, "", "   "])
def test_split_names_empty_input_gives_empty_list(empty):
    assert split_names(empty) == []


# --- prepare --------------------------------------------------------------------------------------

def test_prepare_keeps_only_first_five_actors():
    raw = pd.DataFrame([make_raw(actors="A, B, C, D, E, F, G")])
    assert prepare(raw).loc[0, "actors"] == ["A", "B", "C", "D", "E"]


def test_prepare_counts_suffixed_name_as_one_of_the_five():
    raw = pd.DataFrame([make_raw(actors="Robert Downey, Jr., B, C, D, E, F")])
    assert prepare(raw).loc[0, "actors"] == ["Robert Downey, Jr.", "B", "C", "D", "E"]


def test_prepare_duplicate_actor_counts_once_and_frees_a_slot():
    # The duplicate is removed before taking the first five, so the 6th distinct name moves up.
    raw = pd.DataFrame([make_raw(actors="A, B, A, C, D, E, F")])
    assert prepare(raw).loc[0, "actors"] == ["A", "B", "C", "D", "E"]


def test_to_long_has_no_duplicate_film_role_name_rows():
    raw = pd.DataFrame([make_raw(actors="A, B, A", directors="X, X")])
    roles = to_long(prepare(raw))
    assert not roles.duplicated(["film_id", "role", "name"]).any()


def test_prepare_parses_release_year():
    raw = pd.DataFrame([make_raw(original_release_date="1994-10-14")])
    assert prepare(raw).loc[0, "release_year"] == 1994


@pytest.mark.parametrize("bad_date", [None, "not a date", "14/10/1994"])
def test_prepare_unparseable_date_gives_missing_year(bad_date):
    raw = pd.DataFrame([make_raw(original_release_date=bad_date)])
    assert pd.isna(prepare(raw).loc[0, "release_year"])


def test_prepare_splits_genres_and_directors():
    raw = pd.DataFrame([make_raw(genres="Comedy, Drama", directors="Joel Coen, Ethan Coen")])
    films = prepare(raw)
    assert films.loc[0, "genres"] == ["Comedy", "Drama"]
    assert films.loc[0, "directors"] == ["Joel Coen", "Ethan Coen"]


# --- filters --------------------------------------------------------------------------------------

def test_drop_missing_scores_needs_both_scores():
    films = pd.DataFrame({"critic_score": [70, None, 70, None], "audience_score": [60, 60, None, None]})
    kept = drop_missing_scores(films)
    assert kept.index.tolist() == [0]


def test_drop_missing_year():
    films = pd.DataFrame({"release_year": pd.array([2000, None], dtype="Int64")})
    assert drop_missing_year(films).index.tolist() == [0]


def test_review_filter_critic_boundary_is_inclusive():
    films = pd.DataFrame({"critic_count": [9, 10, 11], "audience_count": [100, 100, 100]})
    assert drop_low_review_counts(films)["critic_count"].tolist() == [10, 11]


def test_review_filter_audience_boundary_is_inclusive():
    films = pd.DataFrame({"critic_count": [20, 20, 20], "audience_count": [49, 50, 51]})
    assert drop_low_review_counts(films)["audience_count"].tolist() == [50, 51]


def test_review_filter_missing_counts_are_dropped():
    films = pd.DataFrame({"critic_count": [None, 20], "audience_count": [100, None]})
    assert drop_low_review_counts(films).empty


# --- run_pipeline ---------------------------------------------------------------------------------

def test_run_pipeline_funnel_adds_up_and_drops_the_right_films():
    raw = pd.DataFrame([
        make_raw(rotten_tomatoes_link="m/keep"),
        make_raw(rotten_tomatoes_link="m/no_audience", audience_rating=None),
        make_raw(rotten_tomatoes_link="m/no_date", original_release_date=None),
        make_raw(rotten_tomatoes_link="m/few_reviews", tomatometer_count=5.0),
        make_raw(rotten_tomatoes_link="m/no_actors", actors=None),  # decision 11: kept
    ])
    films, funnel = run_pipeline(raw)

    assert sorted(films["film_id"]) == ["m/keep", "m/no_actors"]
    assert funnel["dropped"].tolist() == [0, 1, 1, 1]
    assert (funnel["rows_in"] - funnel["dropped"] == funnel["rows_out"]).all()
    assert (funnel["rows_out"].iloc[:-1].values == funnel["rows_in"].iloc[1:].values).all()
    assert funnel["rows_out"].iloc[-1] == len(films)


# --- to_long --------------------------------------------------------------------------------------

def test_to_long_one_row_per_film_role_and_name():
    raw = pd.DataFrame([make_raw(directors="Joel Coen, Ethan Coen", actors="A, B", genres="Comedy")])
    roles = to_long(prepare(raw))
    assert roles.groupby("role").size().to_dict() == {"actor": 2, "director": 2, "genre": 1}
    assert set(roles.loc[roles["role"] == "director", "name"]) == {"Joel Coen", "Ethan Coen"}


def test_to_long_film_without_actors_has_no_actor_rows():
    raw = pd.DataFrame([make_raw(actors=None)])
    roles = to_long(prepare(raw))
    assert "actor" not in set(roles["role"])
    assert not roles["name"].isna().any()
    assert not any(isinstance(n, float) and math.isnan(n) for n in roles["name"])
