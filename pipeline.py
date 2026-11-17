"""Data pipeline for the Rotten Tomatoes actor / director / genre analysis.

Every step is a pure function: it takes a DataFrame and returns a new one, without side effects and
without Streamlit imports. `run_pipeline` chains the steps and records a row funnel (rows in/out per
step and why rows were dropped). The filter thresholds below are the decisions in beslissingslogboek.md.

The second half of the file is the model (log entries 13-24): one ridge regression per score, fitted on
the same films and columns, with bootstrap ranges.

Run `python pipeline.py` to rebuild the files in Data/clean/.
"""
import json
import re
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, sparse

RAW_PATH = Path("Data/raw/RT movies and critics/rotten_tomatoes_movies.csv")
CLEAN_DIR = Path("Data/clean")

TOP_N_ACTORS = 5  # decision 5: first 5 names in billing order
MIN_CRITIC_REVIEWS = 10  # decision 6
MIN_AUDIENCE_RATINGS = 50  # decision 6

# Name parts that were split off by the comma separator, e.g. "Sammy Davis, Jr." -> "Sammy Davis", "Jr."
NAME_SUFFIX = re.compile(r"Jr\.?|Sr\.?|I{1,3}|IV|M\.D\.|Ph\.D\.")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw RT movies file exactly as downloaded."""
    return pd.read_csv(path)


def split_names(text: object) -> list[str]:
    """Split a comma-separated list of names, re-attaching suffixes such as ', Jr.' to the previous name.

    A name listed twice for the same film is kept once, at its first position.
    """
    if not isinstance(text, str):
        return []
    names: list[str] = []
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        if names and NAME_SUFFIX.fullmatch(part):
            names[-1] = f"{names[-1]}, {part}"
        else:
            names.append(part)
    return list(dict.fromkeys(names))  # drop repeated names, keep order


def prepare(raw: pd.DataFrame, top_n_actors: int = TOP_N_ACTORS) -> pd.DataFrame:
    """Keep the needed columns under clear names, parse the release year and split the list columns."""
    films = pd.DataFrame({
        "film_id": raw["rotten_tomatoes_link"],
        "title": raw["movie_title"],
        "release_year": pd.to_datetime(raw["original_release_date"], format="%Y-%m-%d",
                                       errors="coerce").dt.year.astype("Int64"),
        "critic_score": raw["tomatometer_rating"],
        "critic_count": raw["tomatometer_count"],
        "audience_score": raw["audience_rating"],
        "audience_count": raw["audience_count"],
        "content_rating": raw["content_rating"],
        "runtime": raw["runtime"],
        "genres": raw["genres"].map(split_names),
        "directors": raw["directors"].map(split_names),
        "actors": raw["actors"].map(split_names).map(lambda names: names[:top_n_actors]),
    })
    return films


def drop_missing_scores(films: pd.DataFrame) -> pd.DataFrame:
    """Decision 7: keep only films with both a critic and an audience score."""
    return films[films["critic_score"].notna() & films["audience_score"].notna()]


def drop_missing_year(films: pd.DataFrame) -> pd.DataFrame:
    """Decision 8: keep only films with a parseable release year."""
    return films[films["release_year"].notna()]


def drop_low_review_counts(films: pd.DataFrame, min_critic: int = MIN_CRITIC_REVIEWS,
                           min_audience: int = MIN_AUDIENCE_RATINGS) -> pd.DataFrame:
    """Decision 6: keep only films with enough critic reviews and audience ratings."""
    return films[(films["critic_count"] >= min_critic) & (films["audience_count"] >= min_audience)]


FILTER_STEPS = [
    ("Both scores present", "Missing critic or audience score (decision 7)", drop_missing_scores),
    ("Release year present", "No parseable release date (decision 8)", drop_missing_year),
    ("Enough reviews",
     f"Fewer than {MIN_CRITIC_REVIEWS} critic reviews or {MIN_AUDIENCE_RATINGS} audience ratings (decision 6)",
     drop_low_review_counts),
]


def run_pipeline(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the raw data, apply the filter steps in order, and return (films, funnel)."""
    films = prepare(raw)
    funnel = [{"step": "Raw file", "rows_in": len(raw), "rows_out": len(films), "dropped": 0, "reason": ""}]
    for name, reason, step in FILTER_STEPS:
        kept = step(films)
        funnel.append({"step": name, "rows_in": len(films), "rows_out": len(kept),
                       "dropped": len(films) - len(kept), "reason": reason})
        films = kept
    return films.reset_index(drop=True), pd.DataFrame(funnel)


def to_long(films: pd.DataFrame) -> pd.DataFrame:
    """One row per (film, role, name): role is 'actor', 'director' or 'genre' (decision 9: each one counts)."""
    parts = []
    for column, role in [("actors", "actor"), ("directors", "director"), ("genres", "genre")]:
        part = films[["film_id", column]].explode(column).dropna().rename(columns={column: "name"})
        part.insert(1, "role", role)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------
# Model
#
# score = baseline + genres + decade + age rating + runtime group + cast compatibility
#         + director effect(s) + sum of the top-5 actor effects
#
# Ridge regression solved exactly: (X'X + P) b = X'y, where P is a diagonal penalty. With P = 0 this is the
# OLS normal equation. People get penalty lambda (shrinkage n / (n + lambda)); the controls get a tiny
# penalty (effectively unpenalised, entry 18) and the baseline none.
# ---------------------------------------------------------------------------------------------------------
SCORES = ("critic_score", "audience_score")
MIN_FILMS_PERSON = 3  # entry 20: fewer films -> no own effect ("little-known", effect 0)
MIN_OTHER_FILMS_PROFILE = 2  # entry 23: an actor needs >= 2 other films for a genre/era profile
FIRST_DECADE = 1940  # entry 21: earlier films form one group
RUNTIME_BINS = [0, 85, 95, 105, 120, 140, np.inf]  # entry 21, in minutes, bins include the upper edge
RUNTIME_LABELS = ["<=85", "85-95", "95-105", "105-120", "120-140", ">140"]
CONTROL_PENALTY = 1e-4  # entry 18 (A1): controls get lambda * 1e-4, i.e. practically no shrinkage
LAMBDA_GRID = (2, 5, 10, 20, 40, 80)  # entry 19: one lambda for both scores, chosen by cross-validation
CV_FOLDS = 5
N_BOOTSTRAP = 200  # entry 18 (A4)
RANGE_PERCENTILES = (5, 95)  # 90% range
FAN_FACTOR = 0.72  # entry 14: fan score = audience effect - 0.72 * critic effect
COMPAT_FEATURES = ("genre_fit", "cast_genre_similarity", "cast_era_gap")  # entry 23
PERSON_KINDS = ("actor", "director")


def add_categories(films: pd.DataFrame) -> pd.DataFrame:
    """Add the correction categories: decade, age rating and runtime group (missing -> 'unknown')."""
    out = films.copy()
    year = out["release_year"].astype(int)
    out["decade"] = np.where(year < FIRST_DECADE, f"before {FIRST_DECADE}", (year // 10 * 10).astype(str) + "s")
    out["age_rating"] = out["content_rating"].fillna("unknown").astype(str)
    runtime = pd.cut(out["runtime"], RUNTIME_BINS, labels=RUNTIME_LABELS)
    out["runtime_group"] = runtime.cat.add_categories("unknown").fillna("unknown").astype(str)
    return out


def genre_matrix(films: pd.DataFrame, genres: list[str]) -> np.ndarray:
    """0/1 matrix: row = film, column = genre."""
    position = {g: j for j, g in enumerate(genres)}
    m = np.zeros((len(films), len(genres)))
    for i, film_genres in enumerate(films["genres"]):
        for g in film_genres:
            if g in position:
                m[i, position[g]] = 1
    return m


def actor_profiles(films: pd.DataFrame) -> pd.DataFrame:
    """Per actor: number of films, sum of release years and genre counts over the given films (entry 23).

    Only genres and years are used, never scores, so a profile describes what an actor usually plays.
    """
    films = films.reset_index(drop=True)
    genres = sorted(films["genres"].explode().dropna().unique())
    g = genre_matrix(films, genres)
    exploded = films["actors"].explode().dropna()
    rows = exploded.index.to_numpy()
    prof = pd.DataFrame(g[rows], columns=genres)
    prof.insert(0, "year_sum", films["release_year"].astype(float).to_numpy()[rows])
    prof.insert(0, "n_films", 1)
    prof["actor"] = exploded.to_numpy()
    return prof.groupby("actor").sum()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def _mean(values: list[float]) -> float:
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


def compat_features(films: pd.DataFrame, profiles: pd.DataFrame, exclude_self: bool) -> pd.DataFrame:
    """Cast compatibility features per film (entry 23), from the actors' profiles.

    - genre_fit: mean cosine between the film's genres and each actor's genre mix
    - cast_genre_similarity: mean pairwise cosine between the actors' genre mixes
    - cast_era_gap: mean pairwise difference (years) between the actors' average release years

    exclude_self=True when the films are the ones the profiles were built from: the film's own genres and
    year are then removed from each profile, so a film never explains itself. Actors with fewer than
    MIN_OTHER_FILMS_PROFILE (other) films have no profile; a feature stays NaN when no actor (or, for the
    pairwise features, fewer than two actors) has a profile.
    """
    films = films.reset_index(drop=True)
    genres = list(profiles.columns[2:])
    g = genre_matrix(films, genres)
    years = films["release_year"].astype(float).to_numpy()
    row_of = {name: j for j, name in enumerate(profiles.index)}
    mixes = profiles[genres].to_numpy(float)
    counts = profiles["n_films"].to_numpy(float)
    year_sums = profiles["year_sum"].to_numpy(float)
    out = np.full((len(films), len(COMPAT_FEATURES)), np.nan)
    for i, actors in enumerate(films["actors"]):
        vecs, mean_years = [], []
        for actor in actors:
            j = row_of.get(actor)
            if j is None:
                continue
            mix, n, year_sum = mixes[j], counts[j], year_sums[j]
            if exclude_self:
                mix, n, year_sum = mix - g[i], n - 1, year_sum - years[i]
            if n < MIN_OTHER_FILMS_PROFILE:
                continue
            vecs.append(mix)
            mean_years.append(year_sum / n)
        if vecs:
            out[i, 0] = _mean([_cosine(v, g[i]) for v in vecs])
        if len(vecs) >= 2:
            pairs = list(combinations(range(len(vecs)), 2))
            out[i, 1] = _mean([_cosine(vecs[a], vecs[b]) for a, b in pairs])
            out[i, 2] = float(np.mean([abs(mean_years[a] - mean_years[b]) for a, b in pairs]))
    return pd.DataFrame(out, columns=list(COMPAT_FEATURES))


@dataclass
class ModelSpec:
    """Everything needed to turn films into model columns: the column list, actor profiles and the
    mean/sd used to standardise the compatibility features. Built from the training films only."""
    columns: pd.DataFrame  # kind, name, n_films (films in the training data with this column)
    profiles: pd.DataFrame
    compat_mean: pd.Series
    compat_sd: pd.Series


def make_spec(train: pd.DataFrame, min_films: int = MIN_FILMS_PERSON) -> ModelSpec:
    """Decide the model columns from the training films."""
    films = add_categories(train.reset_index(drop=True))
    parts = [pd.DataFrame({"kind": ["baseline"], "name": ["baseline"], "n_films": [len(films)]})]
    for kind, column in [("genre", "genres"), ("decade", "decade"), ("age_rating", "age_rating"),
                         ("runtime", "runtime_group"), ("actor", "actors"), ("director", "directors")]:
        counts = films[column].explode().dropna().value_counts()
        if kind in PERSON_KINDS:
            counts = counts[counts >= min_films]
        counts = counts.sort_index()
        parts.append(pd.DataFrame({"kind": kind, "name": counts.index, "n_films": counts.to_numpy()}))
        if kind == "runtime":  # compatibility columns go between the controls and the people
            parts.append(pd.DataFrame({"kind": "compatibility", "name": list(COMPAT_FEATURES),
                                       "n_films": len(films)}))
    profiles = actor_profiles(films)
    feats = compat_features(films, profiles, exclude_self=True)
    sd = feats.std().replace(0, 1.0).fillna(1.0)
    return ModelSpec(pd.concat(parts, ignore_index=True), profiles, feats.mean().fillna(0.0), sd)


def build_design(films: pd.DataFrame, spec: ModelSpec, exclude_self: bool) -> sparse.csr_matrix:
    """Sparse model matrix: one row per film, columns as in spec.columns.

    Genres, decade, age rating, runtime group, actors and directors are 0/1 columns (entry 18, A3: no
    standardising). Names not in the spec (e.g. an actor with < 3 training films) get no column, i.e.
    effect 0. Compatibility features are standardised with the training mean/sd; a missing feature is set
    to the mean (adds 0).
    """
    films = add_categories(films.reset_index(drop=True))
    col = {(k, n): j for j, (k, n) in enumerate(zip(spec.columns["kind"], spec.columns["name"]))}
    rows, cols, vals = list(range(len(films))), [col[("baseline", "baseline")]] * len(films), [1.0] * len(films)
    for kind, column in [("genre", "genres"), ("decade", "decade"), ("age_rating", "age_rating"),
                         ("runtime", "runtime_group"), ("actor", "actors"), ("director", "directors")]:
        for i, value in films[column].explode().dropna().items():
            j = col.get((kind, value))
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)
    feats = compat_features(films, spec.profiles, exclude_self)
    z = ((feats - spec.compat_mean) / spec.compat_sd).fillna(0.0)
    for name in COMPAT_FEATURES:
        j = col[("compatibility", name)]
        for i, v in enumerate(z[name].to_numpy()):
            if v != 0:
                rows.append(i)
                cols.append(j)
                vals.append(float(v))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(films), len(spec.columns)))


def penalties(columns: pd.DataFrame, lam: float) -> np.ndarray:
    """Diagonal penalty: lambda for people, lambda * CONTROL_PENALTY for controls, 0 for the baseline."""
    p = np.where(columns["kind"].isin(PERSON_KINDS), lam, lam * CONTROL_PENALTY)
    p[(columns["kind"] == "baseline").to_numpy()] = 0.0
    return p


def fit_ridge(X: sparse.spmatrix, Y: np.ndarray, penalty: np.ndarray,
              weights: np.ndarray | None = None) -> np.ndarray:
    """Solve (X'WX + diag(penalty)) b = X'WY exactly. Y may hold several scores (one column each).

    weights = how often each film is counted (used by the bootstrap); None means every film once.
    """
    Xw = X if weights is None else sparse.diags(weights) @ X
    A = (X.T @ Xw).toarray()
    A[np.diag_indices_from(A)] += penalty
    b = np.asarray(Xw.T @ Y).reshape(X.shape[1], -1)
    # Dense Cholesky: ~1 s for ~6,400 columns (sparse solvers were 3-12x slower). The near-zero control
    # penalties make A badly conditioned (rcond ~1e-9), which costs only ~1e-5 points of precision here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        return linalg.solve(A, b, assume_a="pos")


def predict(X: sparse.spmatrix, coef: np.ndarray) -> np.ndarray:
    """Predicted scores, capped at 0-100 (entry 16; limitation B3)."""
    return np.clip(X @ coef, 0, 100)


CV_VARIANTS = {  # which column kinds each compared model uses
    "controls only": {"baseline", "genre", "decade", "age_rating", "runtime"},
    "controls + people": {"baseline", "genre", "decade", "age_rating", "runtime", "actor", "director"},
    "full model": {"baseline", "genre", "decade", "age_rating", "runtime", "actor", "director", "compatibility"},
}


def cross_validate(films: pd.DataFrame, lambdas=LAMBDA_GRID, folds: int = CV_FOLDS, seed: int = 0) -> pd.DataFrame:
    """Held-out mean absolute error per fold, lambda, model variant and score.

    Everything (columns, profiles, standardisation) is rebuilt from the training part of each fold, so the
    held-out films are truly unseen.
    """
    films = films.reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(films))
    rows = []
    for k, val_idx in enumerate(np.array_split(order, folds)):
        train, val = films.drop(index=val_idx), films.loc[val_idx]
        spec = make_spec(train)
        X_tr, X_va = build_design(train, spec, exclude_self=True), build_design(val, spec, exclude_self=False)
        Y_tr, Y_va = train[list(SCORES)].to_numpy(float), val[list(SCORES)].to_numpy(float)
        for lam in lambdas:
            pen = penalties(spec.columns, lam)
            for variant, kinds in CV_VARIANTS.items():
                keep = spec.columns["kind"].isin(kinds).to_numpy()
                coef = fit_ridge(X_tr[:, keep], Y_tr, pen[keep])
                err = np.abs(predict(X_va[:, keep], coef) - Y_va).mean(axis=0)
                for s, e in zip(SCORES, err):
                    rows.append({"fold": k, "lambda": lam, "variant": variant, "score": s, "mae": e})
    return pd.DataFrame(rows)


def choose_lambda(cv: pd.DataFrame, films: pd.DataFrame) -> float:
    """Entry 19: one lambda for both scores. Each score's error is divided by that score's standard
    deviation (critic scores spread more), then the two are averaged; the lowest average wins."""
    full = cv[cv["variant"] == "full model"].copy()
    full["scaled"] = full["mae"] / full["score"].map(films[list(SCORES)].std())
    return float(full.groupby("lambda")["scaled"].mean().idxmin())


def fit_with_bootstrap(films: pd.DataFrame, lam: float, n_boot: int = N_BOOTSTRAP, seed: int = 0):
    """Fit the final model on all films, plus n_boot bootstrap refits (resampling films, lambda fixed).

    Returns (spec, estimate, draws): estimate has shape (columns, scores), draws (n_boot, columns, scores).
    The columns and compatibility features stay fixed; only how often each film counts changes.
    """
    films = films.reset_index(drop=True)
    spec = make_spec(films)
    X = build_design(films, spec, exclude_self=True)
    Y = films[list(SCORES)].to_numpy(float)
    pen = penalties(spec.columns, lam)
    estimate = fit_ridge(X, Y, pen)
    rng = np.random.default_rng(seed)
    n = len(films)
    draws = np.empty((n_boot, X.shape[1], len(SCORES)), dtype=np.float32)
    for b in range(n_boot):
        weights = rng.multinomial(n, np.full(n, 1 / n)).astype(float)  # = drawing n films with replacement
        draws[b] = fit_ridge(X, Y, pen, weights)
    return spec, estimate, draws


def summarise(spec: ModelSpec, estimate: np.ndarray, draws: np.ndarray) -> pd.DataFrame:
    """One row per model column: estimate and 90% range per score; for people also the fan score."""
    lo, hi = RANGE_PERCENTILES
    out = spec.columns.copy()
    for k, s in enumerate(SCORES):
        label = s.replace("_score", "")
        out[f"{label}_effect"] = estimate[:, k]
        out[f"{label}_low"] = np.percentile(draws[:, :, k], lo, axis=0)
        out[f"{label}_high"] = np.percentile(draws[:, :, k], hi, axis=0)
    fan_draws = draws[:, :, 1] - FAN_FACTOR * draws[:, :, 0]
    person = out["kind"].isin(PERSON_KINDS)
    out["fan_score"] = np.where(person, estimate[:, 1] - FAN_FACTOR * estimate[:, 0], np.nan)
    out["fan_low"] = np.where(person, np.percentile(fan_draws, lo, axis=0), np.nan)
    out["fan_high"] = np.where(person, np.percentile(fan_draws, hi, axis=0), np.nan)
    return out


def save_model(spec: ModelSpec, coefficients: pd.DataFrame, draws: np.ndarray, info: dict,
               out_dir: Path = CLEAN_DIR) -> None:
    """Write what the app needs: coefficients, bootstrap draws, actor profiles and model settings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(out_dir / "model_coefficients.csv", index=False)
    np.savez_compressed(out_dir / "model_bootstrap.npz", draws=draws)
    spec.profiles.to_csv(out_dir / "actor_profiles.csv")
    info = {**info, "compat_mean": spec.compat_mean.to_dict(), "compat_sd": spec.compat_sd.to_dict()}
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_model(out_dir: Path = CLEAN_DIR):
    """Read the saved model back: (spec, coefficients, draws, info)."""
    coefficients = pd.read_csv(out_dir / "model_coefficients.csv", keep_default_na=False,
                               na_values={c: [""] for c in ["fan_score", "fan_low", "fan_high"]})
    draws = np.load(out_dir / "model_bootstrap.npz")["draws"]
    profiles = pd.read_csv(out_dir / "actor_profiles.csv", index_col="actor", keep_default_na=False)
    info = json.loads((out_dir / "model_info.json").read_text(encoding="utf-8"))
    spec = ModelSpec(coefficients[["kind", "name", "n_films"]], profiles,
                     pd.Series(info["compat_mean"]), pd.Series(info["compat_sd"]))
    return spec, coefficients, draws, info


MIN_SHARED_FILMS = 5  # entry 24: pairs need >= 5 shared films ...
MIN_FILMS_APART = 3  # ... and each partner >= 3 films without the other


def out_of_fold_residuals(films: pd.DataFrame, lam: float, folds: int = CV_FOLDS, seed: int = 0) -> np.ndarray:
    """Actual minus predicted score per film, each film predicted by a model that never saw it."""
    films = films.reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(films))
    resid = np.empty((len(films), len(SCORES)))
    for val_idx in np.array_split(order, folds):
        train, val = films.drop(index=val_idx), films.loc[val_idx]
        spec = make_spec(train)
        coef = fit_ridge(build_design(train, spec, exclude_self=True), train[list(SCORES)].to_numpy(float),
                         penalties(spec.columns, lam))
        pred = predict(build_design(val, spec, exclude_self=False), coef)
        resid[val_idx] = val[list(SCORES)].to_numpy(float) - pred
    return resid


def save_honest_predictions(films: pd.DataFrame, resid: np.ndarray, out_dir: Path = CLEAN_DIR) -> pd.DataFrame:
    """Per film the prediction of a model that never saw it (actual - out-of-fold residual), for the app."""
    films = films.reset_index(drop=True)
    out = pd.DataFrame({"film_id": films["film_id"]})
    for k, s in enumerate(SCORES):
        out[s.replace("_score", "_honest")] = films[s].to_numpy(float) - resid[:, k]
    out.to_csv(out_dir / "film_predictions.csv", index=False)
    return out


def collaborations(films: pd.DataFrame, resid: np.ndarray, n_resample: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Standout recurring collaborations (entry 24, exploratory): mean out-of-fold residual of shared films.

    Pairs: actor x actor and actor x director, >= MIN_SHARED_FILMS shared films, each partner
    >= MIN_FILMS_APART films without the other, no self-direction. Pairs with exactly the same shared films
    (e.g. a trio that always appears together) are merged into one group. The 90% range resamples only the
    shared films, so it ignores model uncertainty and is too narrow (see the caveats in the app).
    """
    films = films.reset_index(drop=True)
    films_of: dict[tuple[str, str], set[int]] = {}
    for i, (actors, directors) in enumerate(zip(films["actors"], films["directors"])):
        for a in actors:
            films_of.setdefault(("actor", a), set()).add(i)
        for d in directors:
            films_of.setdefault(("director", d), set()).add(i)
    shared: dict[tuple, set[int]] = {}
    for i, (actors, directors) in enumerate(zip(films["actors"], films["directors"])):
        for a, b in combinations(sorted(actors), 2):
            shared.setdefault((("actor", a), ("actor", b)), set()).add(i)
        for a in actors:
            for d in directors:
                if a != d:  # no self-direction
                    shared.setdefault((("actor", a), ("director", d)), set()).add(i)
    groups: dict[frozenset, set] = {}
    for (p1, p2), idx in shared.items():
        if (len(idx) >= MIN_SHARED_FILMS and len(films_of[p1] - idx) >= MIN_FILMS_APART
                and len(films_of[p2] - idx) >= MIN_FILMS_APART):
            groups.setdefault(frozenset(idx), set()).update([p1, p2])
    rng = np.random.default_rng(seed)
    lo, hi = RANGE_PERCENTILES
    rows = []
    for idx, people in groups.items():
        idx = sorted(idx)
        row = {"people": ", ".join(f"{n} (director)" if k == "director" else n for k, n in sorted(people, key=lambda x: (x[0], x[1]))),
               "shared_films": len(idx),
               "example_titles": "; ".join(films.loc[idx].sort_values("release_year")["title"].head(4))}
        for k, s in enumerate(SCORES):
            label = s.replace("_score", "")
            r = resid[idx, k]
            boot = r[rng.integers(0, len(idx), size=(n_resample, len(idx)))].mean(axis=1)
            row[f"{label}_surprise"] = r.mean()
            row[f"{label}_low"] = np.percentile(boot, lo)
            row[f"{label}_high"] = np.percentile(boot, hi)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("critic_surprise", ascending=False, ignore_index=True)


def run_model(films: pd.DataFrame, out_dir: Path = CLEAN_DIR) -> pd.DataFrame:
    """Cross-validate lambda, fit the final model with bootstrap ranges, save it, and return the CV table."""
    t0 = time.time()
    cv = cross_validate(films)
    lam = choose_lambda(cv, films)
    print(f"\nCross-validation ({CV_FOLDS} folds, {time.time() - t0:.0f} s): chosen lambda = {lam:g}")
    table = cv.groupby(["lambda", "variant", "score"])["mae"].mean().unstack("score").round(2)
    print(table.to_string())
    t0 = time.time()
    spec, estimate, draws = fit_with_bootstrap(films, lam)
    coefficients = summarise(spec, estimate, draws)
    info = {"lambda": lam, "n_films": len(films), "n_bootstrap": N_BOOTSTRAP, "fan_factor": FAN_FACTOR,
            "range_percentiles": list(RANGE_PERCENTILES), "min_films_person": MIN_FILMS_PERSON}
    save_model(spec, coefficients, draws, info, out_dir)
    cv.to_csv(out_dir / "model_cv.csv", index=False)
    n_people = coefficients["kind"].value_counts().reindex(list(PERSON_KINDS)).to_dict()
    print(f"\nFinal model + {N_BOOTSTRAP} bootstrap refits ({time.time() - t0:.0f} s): {len(spec.columns)} columns, "
          f"people with an own effect: {n_people}")
    resid = out_of_fold_residuals(films, lam)
    save_honest_predictions(films, resid, out_dir)
    collabs = collaborations(films, resid)
    collabs.to_csv(out_dir / "collaborations.csv", index=False)
    print(f"Standout recurring collaborations: {len(collabs)} groups/pairs written")
    return cv


def main() -> None:
    films, funnel = run_pipeline(load_raw())
    roles = to_long(films)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    films.drop(columns=["genres", "directors", "actors"]).to_csv(CLEAN_DIR / "films.csv", index=False)
    roles.to_csv(CLEAN_DIR / "film_roles.csv", index=False)
    funnel.to_csv(CLEAN_DIR / "funnel.csv", index=False)
    print(funnel.to_string(index=False))
    print(f"\nWrote {len(films)} films and {len(roles)} film-role rows to {CLEAN_DIR}/")
    run_model(films)


if __name__ == "__main__":
    main()
