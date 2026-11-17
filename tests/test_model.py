"""Tests for the model part of pipeline.py, on small hand-made film tables."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

import pipeline as p


def film(**overrides) -> dict:
    """One prepared film (the layout run_pipeline returns), with harmless defaults."""
    base = {"film_id": "m/x", "title": "X", "release_year": 2005, "critic_score": 50.0, "critic_count": 20,
            "audience_score": 50.0, "audience_count": 100, "content_rating": "R", "runtime": 100.0,
            "genres": ["Drama"], "directors": ["Dir"], "actors": ["A", "B"]}
    base.update(overrides)
    return base


def frame(*films: dict) -> pd.DataFrame:
    return pd.DataFrame(list(films))


# --- corrections -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("year, decade", [(1925, "before 1940"), (1939, "before 1940"), (1940, "1940s"),
                                          (2019, "2010s")])
def test_decade_groups(year, decade):
    assert p.add_categories(frame(film(release_year=year)))["decade"].iloc[0] == decade


@pytest.mark.parametrize("minutes, group", [(85, "<=85"), (86, "85-95"), (105, "95-105"), (140, "120-140"),
                                            (141, ">140"), (np.nan, "unknown")])
def test_runtime_groups_include_upper_edge(minutes, group):
    assert p.add_categories(frame(film(runtime=minutes)))["runtime_group"].iloc[0] == group


def test_missing_age_rating_is_unknown():
    assert p.add_categories(frame(film(content_rating=None)))["age_rating"].iloc[0] == "unknown"


# --- ridge solver ------------------------------------------------------------------------------------------

def test_fit_ridge_matches_sklearn():
    """Same penalty on every column and none on the baseline = sklearn Ridge with an intercept."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(50, 4))
    y = Z @ [1.0, -2.0, 0.5, 3.0] + rng.normal(size=50)
    X = sparse.csr_matrix(np.column_stack([np.ones(50), Z]))
    coef = p.fit_ridge(X, y[:, None], np.array([0, 3.0, 3.0, 3.0, 3.0]))[:, 0]
    sk = Ridge(alpha=3.0).fit(Z, y)
    assert coef[0] == pytest.approx(sk.intercept_)
    assert coef[1:] == pytest.approx(sk.coef_)


@pytest.mark.parametrize("n, lam", [(3, 10), (20, 10), (5, 2)])
def test_shrinkage_is_n_over_n_plus_lambda(n, lam):
    """One actor in n films that all score 10 points higher: ridge keeps n / (n + lambda) of the 10."""
    x = np.zeros(40)
    x[:n] = 1
    y = 10 * x
    coef = p.fit_ridge(sparse.csr_matrix(x[:, None]), y[:, None], np.array([lam]))
    assert coef[0, 0] == pytest.approx(10 * n / (n + lam))


def test_weights_equal_duplicated_films():
    """Counting a film twice (bootstrap weight 2) is the same as having it twice in the data."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(10, 3))
    y = rng.normal(size=10)
    w = np.array([1, 2, 0, 1, 3, 1, 1, 0, 1, 1], dtype=float)
    pen = np.array([0.5, 0.5, 0.5])
    weighted = p.fit_ridge(sparse.csr_matrix(Z), y[:, None], pen, w)
    rep = np.repeat(np.arange(10), w.astype(int))
    duplicated = p.fit_ridge(sparse.csr_matrix(Z[rep]), y[rep][:, None], pen)
    assert weighted == pytest.approx(duplicated)


def test_penalties_per_kind():
    cols = pd.DataFrame({"kind": ["baseline", "genre", "compatibility", "actor", "director"]})
    assert list(p.penalties(cols, 10)) == pytest.approx([0, 10 * p.CONTROL_PENALTY, 10 * p.CONTROL_PENALTY, 10, 10])


def test_predictions_capped_at_0_100():
    X = sparse.csr_matrix(np.array([[1.0], [1.0], [1.0]]))
    assert list(p.predict(X, np.array([[120.0]]))[:, 0]) == [100, 100, 100]
    assert list(p.predict(X, np.array([[-5.0]]))[:, 0]) == [0, 0, 0]


# --- columns -----------------------------------------------------------------------------------------------

def test_people_need_three_films_for_a_column():
    films = frame(*[film(actors=["Often"], directors=["D3"]) for _ in range(3)],
                  *[film(actors=["Twice"], directors=["D2"]) for _ in range(2)])
    names = set(p.make_spec(films).columns.query("kind in ['actor', 'director']")["name"])
    assert names == {"Often", "D3"}


def test_little_known_actor_adds_nothing():
    """An actor below the film threshold gets no column: the row is the same as without that actor."""
    films = frame(*[film(actors=["Often"]) for _ in range(3)])
    spec = p.make_spec(films)
    with_unknown = p.build_design(frame(film(actors=["Often", "Newcomer"])), spec, exclude_self=False)
    without = p.build_design(frame(film(actors=["Often"])), spec, exclude_self=False)
    assert (with_unknown != without).nnz == 0


def test_multi_genre_film_gets_every_genre():
    films = frame(film(genres=["Comedy", "Drama"]), film(genres=["Drama"]))
    spec = p.make_spec(films)
    X = p.build_design(films, spec, exclude_self=False).toarray()
    genre_cols = spec.columns.index[spec.columns["kind"] == "genre"]
    assert X[0, genre_cols].sum() == 2 and X[1, genre_cols].sum() == 1


# --- cast compatibility ------------------------------------------------------------------------------------

@pytest.fixture
def compat_films():
    """A plays comedies, B plays mostly dramas; film 0 has both."""
    return frame(
        film(genres=["Comedy"], actors=["A", "B"], release_year=2000),
        film(genres=["Comedy"], actors=["A"], release_year=2002),
        film(genres=["Comedy"], actors=["A"], release_year=2004),
        film(genres=["Drama"], actors=["B"], release_year=1980),
        film(genres=["Drama"], actors=["B"], release_year=1990),
    )


def test_compat_leaves_the_film_itself_out(compat_films):
    profiles = p.actor_profiles(compat_films)
    feats = p.compat_features(compat_films, profiles, exclude_self=True).iloc[0]
    # Other films: A = 2 comedies (fit 1), B = 2 dramas (fit 0) -> genre fit 0.5; no shared genre -> 0
    assert feats["genre_fit"] == pytest.approx(0.5)
    assert feats["cast_genre_similarity"] == pytest.approx(0.0)
    # A's other films average 2003, B's 1985
    assert feats["cast_era_gap"] == pytest.approx(18.0)


def test_compat_without_leaving_out_uses_all_films(compat_films):
    profiles = p.actor_profiles(compat_films)
    feats = p.compat_features(compat_films.iloc[[0]], profiles, exclude_self=False).iloc[0]
    # A = 3 comedies (fit 1); B = 1 comedy + 2 dramas (fit 1/sqrt(5)); similarity = 1/sqrt(5)
    assert feats["genre_fit"] == pytest.approx((1 + 1 / np.sqrt(5)) / 2)
    assert feats["cast_genre_similarity"] == pytest.approx(1 / np.sqrt(5))


def test_profile_needs_two_other_films():
    films = frame(film(actors=["A", "B"]), film(actors=["A", "B"]))  # each has 1 other film
    feats = p.compat_features(films, p.actor_profiles(films), exclude_self=True)
    assert feats.isna().all().all()


def test_compat_does_not_use_scores(compat_films):
    changed = compat_films.assign(critic_score=0.0, audience_score=100.0)
    a = p.compat_features(compat_films, p.actor_profiles(compat_films), exclude_self=True)
    b = p.compat_features(changed, p.actor_profiles(changed), exclude_self=True)
    pd.testing.assert_frame_equal(a, b)


# --- whole model -------------------------------------------------------------------------------------------

def synthetic_films(n=300, star_films=20, bonus=20.0, seed=0) -> pd.DataFrame:
    """Random casts from a pool of 30 actors; 'Star' raises the critic score by `bonus` in their films."""
    rng = np.random.default_rng(seed)
    pool = [f"Actor {k}" for k in range(30)]
    rows = []
    for i in range(n):
        cast = list(rng.choice(pool, 3, replace=False)) + (["Star"] if i < star_films else [])
        noise = rng.normal(0, 5)
        rows.append(film(film_id=f"m/{i}", actors=cast, directors=[f"Dir {i % 10}"],
                         genres=[rng.choice(["Drama", "Comedy"])], release_year=int(rng.integers(1950, 2020)),
                         critic_score=50 + noise + (bonus if i < star_films else 0), audience_score=50 + noise))
    return frame(*rows)


def test_planted_actor_effect_is_recovered():
    films = synthetic_films()
    spec = p.make_spec(films)
    X = p.build_design(films, spec, exclude_self=True)
    lam = 10
    coef = p.fit_ridge(X, films[list(p.SCORES)].to_numpy(float), p.penalties(spec.columns, lam))
    star = spec.columns.index[(spec.columns["kind"] == "actor") & (spec.columns["name"] == "Star")][0]
    others = spec.columns.index[(spec.columns["kind"] == "actor") & (spec.columns["name"] != "Star")]
    assert coef[star, 0] == pytest.approx(20 * 20 / (20 + lam), abs=2.5)  # shrunk towards 0, about 13.3
    assert np.abs(coef[others, 0]).max() < 3  # nobody else gets a big critic effect
    assert abs(coef[star, 1]) < 3  # and Star has no audience effect


def test_bootstrap_ranges_and_fan_score():
    films = synthetic_films()
    spec, estimate, draws = p.fit_with_bootstrap(films, lam=10, n_boot=30)
    out = p.summarise(spec, estimate, draws)
    star = out[(out["kind"] == "actor") & (out["name"] == "Star")].iloc[0]
    assert star["critic_low"] > 0  # clearly positive with critics
    assert star["critic_low"] <= star["critic_effect"] <= star["critic_high"]
    assert star["fan_score"] == pytest.approx(star["audience_effect"] - 0.72 * star["critic_effect"])
    assert star["fan_high"] < 0  # relatively more liked by critics than by the audience
    assert out.loc[out["kind"] == "genre", "fan_score"].isna().all()  # fan score only for people


def test_save_and_load_roundtrip(tmp_path):
    films = synthetic_films(n=120, star_films=10)
    spec, estimate, draws = p.fit_with_bootstrap(films, lam=10, n_boot=5)
    out = p.summarise(spec, estimate, draws)
    p.save_model(spec, out, draws, {"lambda": 10}, tmp_path)
    spec2, out2, draws2, info = p.load_model(tmp_path)
    new_film = frame(film(actors=["Star", "Actor 1"], genres=["Comedy"]))
    X1 = p.build_design(new_film, spec, exclude_self=False)
    X2 = p.build_design(new_film, spec2, exclude_self=False)
    assert (X1 != X2).nnz == 0
    assert np.allclose(draws, draws2) and info["lambda"] == 10


# --- recurring collaborations ------------------------------------------------------------------------------

def collab_films() -> pd.DataFrame:
    """Trio X, Y, Z shares 5 films; each also has 3 films alone. 'Self' acts in and directs 5 films."""
    together = [film(film_id=f"t{i}", actors=["X", "Y", "Z"], directors=["Dir A"]) for i in range(5)]
    alone = [film(film_id=f"{a}{i}", actors=[a], directors=["Dir B"]) for a in "XYZ" for i in range(3)]
    self_directed = [film(film_id=f"s{i}", actors=["Self"], directors=["Self"]) for i in range(5)]
    return frame(*together, *alone, *self_directed)


def test_collaborations_merge_groups_and_skip_self_direction():
    films = collab_films()
    resid = np.zeros((len(films), 2))
    resid[:5] = 10.0  # the shared films score 10 points above prediction
    out = p.collaborations(films, resid, n_resample=50)
    assert list(out["people"]) == ["X, Y, Z"]  # three pairs with the same films -> one row; no 'Self' pair
    assert out["shared_films"].iloc[0] == 5 and out["critic_surprise"].iloc[0] == pytest.approx(10.0)


def test_collaborations_need_films_apart():
    films = collab_films()
    films = films[~films["film_id"].isin(["Z0", "Z1"])]  # Z now has only 1 film without X and Y
    out = p.collaborations(films, np.zeros((len(films), 2)), n_resample=10)
    assert list(out["people"]) == ["X, Y"]
