"""Data pipeline for the Rotten Tomatoes actor / director / genre analysis.

Every step is a pure function: it takes a DataFrame and returns a new one, without side effects and
without Streamlit imports. `run_pipeline` chains the steps and records a row funnel (rows in/out per
step and why rows were dropped). The filter thresholds below are the decisions in beslissingslogboek.md.

Run `python pipeline.py` to rebuild the files in Data/clean/.
"""
import re
from pathlib import Path

import pandas as pd

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
    """Split a comma-separated list of names, re-attaching suffixes such as ', Jr.' to the previous name."""
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
    return names


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


def main() -> None:
    films, funnel = run_pipeline(load_raw())
    roles = to_long(films)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    films.drop(columns=["genres", "directors", "actors"]).to_csv(CLEAN_DIR / "films.csv", index=False)
    roles.to_csv(CLEAN_DIR / "film_roles.csv", index=False)
    funnel.to_csv(CLEAN_DIR / "funnel.csv", index=False)
    print(funnel.to_string(index=False))
    print(f"\nWrote {len(films)} films and {len(roles)} film-role rows to {CLEAN_DIR}/")


if __name__ == "__main__":
    main()
