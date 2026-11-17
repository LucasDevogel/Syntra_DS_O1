# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Syntra Data Scientist course, assignment 1. Using Rotten Tomatoes scores combined with film metadata, explain
what drives film ratings. Every question is answered **separately for two outcomes**:

- **Critic score** (Rotten Tomatoes critics)
- **User score** (Rotten Tomatoes audience score)

Confirmed: "score" always means a Rotten Tomatoes score.

Interpretation (agreed 2026-11-16): both RT scores are **percentages of positive reviews/ratings**, not
average grades. Measured effects are therefore about **likeability** (how many people liked a film), not
about film quality. Always phrase results that way, e.g. "+6 points = 6 percentage points more critics
positive". This fits the research aim of telling **fan favourites** (high audience scores) apart from
**critically acclaimed** actors and directors.

Questions:

1. **Individual actor effect**: isolate the effect a single actor has on critic and user scores, after
   accounting for director, genre and other confounders.
2. **Individual director effect**: the same for directors.
3. **Synergy**: do specific combinations (actor x actor, actor x director) score higher than their individual
   effects predict? This is an interaction effect on top of the individual effects.
4. **Underminers**: which actors or directors have a clearly negative individual effect, i.e. pull scores
   down on their own?
5. **Genre effect**: how much does genre alone shift critic and user scores?

Later stage: **audience vs. critic favourites**: are some actors (and directors) significantly more liked by
the public than by critics, or the other way round? This needs both scores on the same films, so any data
source used must contain the RT audience score as well as the critic score.

## Assignment requirements (Project 1, Syntra module M1)

Background: `Project 1 - demo-case verkeersveiligheid.pdf` describes a 7-step process (design, spec prompt,
pipeline, CLAUDE.md, git, tests, deploy). The scope has since been **scaled down**: the steps are guidance,
not a strict checklist, and design and testing do not need to be rigorous.

What the project must do:

1. **Present the research question.**
2. **Present the answer** in whatever form fits the question best (a Streamlit app is the intended form).
3. **Briefly explain the path from raw data to result.** Focus on the **non-trivial** decisions (matching,
   thresholds, model choice, confounding), not on routine cleaning.

Deliverables (all still required):

- GitHub repo link containing `pipeline.py`, `app.py`, tests, `CLAUDE.md` and `beslissingslogboek.md`
- A working, public Streamlit URL (not localhost)
- The original design (photo or digital file; may be lightweight)

Keep in mind for the no-AI conversation at delivery: the student must be able to explain every number and
choice in the app without AI. The decision log is a memory aid.

## Final deliverable: Streamlit app

`app.py` has three parts:

1. **Method** (brief): the research question, and how raw data became the result: a short row funnel
   (rows kept and dropped per step, and why) and the key non-trivial decisions. Pipeline functions live in
   `pipeline.py` (no Streamlit imports).
2. **Effect lookup**: select an actor or director and see their estimated average effect on the critic score
   and the user score, with an uncertainty range and the number of films the estimate is based on.
3. **Score estimator**: select one director, a list of actors and a genre, and get an estimated critic
   score and user score for that combination, with an uncertainty range. The estimate comes from the same
   model as the effects (baseline + genre + director + actors + any known synergy terms). People with too few
   films contribute (close to) zero effect, and the app must say so instead of hiding it.

Deployment follows the course rules: a `requirements.txt` in the app repo with only what the app needs, and
no `environment.yml` in the app repo. The data the app needs must be available on Streamlit Cloud.
GitHub rejects files over 100 MB, so check raw file sizes before deciding what goes into the repo.

## Data source

**Single source**: *Rotten Tomatoes movies and critic reviews dataset* (Kaggle, stefanoleone992):
https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
No joins with other datasets are needed.

History: the project first used the Mendeley RT file (`movie_info.csv`) plus the Kaggle *Movies Dataset*
(rounakbanik) for cast and genre, matched on title + release date. These were dropped on 2026-11-16 for data
completeness (no review counts, fewer films with both scores, error-prone matching). See decision log
entries 1-4. The matching rules from that period no longer apply.

Files in `Data/raw/RT movies and critics/` (inspected 2026-11-16):

| File | Rows | Notes |
|---|---|---|
| `rotten_tomatoes_movies.csv` (17 MB) | 17,712 films, key `rotten_tomatoes_link` (unique, e.g. `m/0814255`) | **The data source.** See columns below. |
| `rotten_tomatoes_critic_reviews.csv` (226 MB) | 1,130,017 individual critic reviews (critics only, no audience data) | **Not used** (decided 2026-11-16). Its Fresh share per film reproduces `tomatometer_rating`, so it adds nothing for the questions. |

Key columns of `rotten_tomatoes_movies.csv`:

| Column | Meaning | Notes |
|---|---|---|
| `tomatometer_rating` | **critic score**, 0-100 | missing for 44 films |
| `tomatometer_count` | number of critic reviews | min 5 (RT shows no score below 5), median 28; 3,314 films < 10 |
| `audience_rating` | **user score**, 0-100 | missing for 296 films |
| `audience_count` | number of audience ratings | median 4,277; 565 films < 50 |
| `genres` | comma-separated, 21 RT genres | films often have several genres |
| `directors` | comma-separated | 1 director for 15,680 films, 2+ for ~1,800; missing for 194 |
| `actors` | comma-separated | median 19 names, p90 52. **No explicit billing order** (first names look like the leads, not verified). Name variants occur (e.g. *Sam Jones III* / *Sam J. Jones*). Missing for 352 |
| `original_release_date` | release date | 1914-2020; missing for 1,166 |

17,407 films have both scores. The `*_status` columns are labels derived from the scores and not needed.

Exploration findings (2026-11-16):

- 15,996 films have both scores, director, actors, genre and release year (16,942 if year is not required).
- **Duplicates**: none in practice. The only same title + year pairs (*Hamlet* 2000, *Noise* 2007) are
  different films with different directors.
- **Cast order** looks like billing order: only 1.0% of lists are alphabetical, and for 8 well-known films
  (*The Godfather*, *Pulp Fiction*, *Forrest Gump*, ...) the first names are the leads.
- Films per actor (films with both scores): first 5 names → 3,548 actors with >= 5 films, 1,625 with >= 10;
  whole cast → 16,828 with >= 5 films. Directors: 877 with >= 5 films, 223 with >= 10; 1,815 films have 2+.
- `(II)`-style suffixes are RT's way of telling apart different people with the same name, so they are
  not variants. True spelling variants (*Sam Jones III* / *Sam J. Jones*) exist but were not quantified.
- Genres: 1-7 per film (mostly 1-3). `Documentary`, `Special Interest`, `Television` are also genres.
- Value check (2026-11-17): both scores are whole numbers within 0-100 in the raw file and after the pipeline;
  review counts are whole and non-negative. After the pipeline 420 films have a critic score of exactly 100
  and 107 of exactly 0 (ceiling/floor); audience scores hit 0 or 100 only 6 times.
- Critic vs. audience correlation 0.65. **Older films score much higher with critics** (pre-1970 mean critic
  80.9 vs. audience 72.2): likely survivorship (mostly classics from that era are on RT). Release year is a
  confounder.

Agreed filters and definitions (decision log entries 5-9, 2026-11-16):

1. **Actors**: only the first 5 names in `actors` (billing order).
2. **Minimum reviews**: `tomatometer_count >= 10` and `audience_count >= 50`.
3. **Both scores required**: drop films missing `tomatometer_rating` or `audience_rating`.
4. **Release year required**: drop films without a parseable `original_release_date`.
5. **Multiple genres / directors**: every genre and every director listed for a film counts for that film.
6. **Duplicate names count once** (entry 12): repeats are removed from the whole list before the top 5 is
   taken; same for directors and genres.
7. **Films without actors or director are kept** (entry 11): they still inform genre/year (and actor or
   director) effects. No merge with other datasets to fill the gaps.

The pipeline applies these as separate steps and reports rows in/out per step (the row funnel). Counts in
the log were computed per filter in isolation; the sequential funnel numbers will differ slightly.

## Modelling (decisions in progress, 2026-11-17)

Agreed so far:

- **Model type** (log entry 13): ridge regression, one model per score (critic, audience), all effects
  estimated together: score = baseline + genres + directors + sum of top-5 actor effects + corrections.
  Demo on critic score (80/20 split): mean abs. error 18.4 vs. 19.4 without people, 24.8 with OLS.
- **Implementation choices A1-A4** (agreed 2026-11-17):
  - A1: only actor and director coefficients are penalised; corrections (genre, decade, age rating) are
    effectively unpenalised (scale those columns up).
  - A2: one shared λ for both scores, chosen by cross-validation on both together, so the fan-favourite
    comparison (audience effect − critic effect) is fair.
  - A3: plain 0/1 indicator columns, no standardising (keeps the n/(n+λ) shrinkage intuition).
  - A4: bootstrap with 200 resamples of films, 90% percentile ranges, λ fixed during the bootstrap.
- **Optional extension** (agreed 2026-11-17): actor strength per genre, done like synergy as a second step
  (mean residual of the main model for actor-genre combinations with >= 5 films), not as interaction terms
  in the main model. 4,786 actor-genre combinations have >= 5 films; 1,237 actors have >= 5 in 2+ genres.

- **Minimum films** (decision 2, agreed 2026-11-17): actors and directors with >= 3 films get their own
  coefficient; everyone below counts as 0 ("little-known"). App: search and score estimator use everyone
  with >= 3 films (always showing film count and range); rankings (boosters, underminers, fan favourites)
  only show people with >= 5 films. Evidence (ridge, λ = 10, 5 splits, MAE critic/audience): >= 1: 18.57/13.88,
  >= 2: 18.53/13.87, >= 3: 18.56/13.92, >= 5: 18.71/13.98, >= 10: 19.00/14.12, no people: 19.76/14.47.

- **Corrections / confounders** (decision 3, agreed 2026-11-17): decade (years before 1940 merged into one
  group), age rating (`content_rating`), runtime in 6 categories (<= 85, 85-95, 95-105, 105-120, 120-140,
  > 140 min) plus "unknown" (80 films). Evidence (MAE critic/audience, 5 splits): genre only 19.16/14.25,
  + decade 18.70/13.92, + age rating 18.57/13.91, + runtime 18.36/13.51. Decade shifts classic people most
  (Hitchcock critic +7.1 → +1.8, Cary Grant +6.1 → +2.4); age rating and runtime change effects little
  (median 0.3 points). Age rating adds little on top of decade + genre (NR = 65% of pre-1968 films).
  Phrase as "improves prediction on unseen films", not "significant" (no significance tests were run).
- **Score estimator inputs**: decade, age rating and runtime are optional inputs with visible defaults
  (2010s, R, 95-105 min). Runtime via a slider in minutes that shows which category is used (the model
  works in categories, so the prediction changes in steps).

- **Fan score** (decision 4, log entry 14): every person has a critic effect and an audience effect from two
  separate models (same films, columns and λ). Fan score = audience effect − 0.72 × critic effect, in audience
  points (0.72 = sd audience 20.0 / sd critic 27.7). Positive = fan favourite, negative = critics' favourite;
  label only when the 90% range from the paired bootstrap lies entirely above/below 0, else "no clear
  difference". Effects are relative to similar films (genre, decade, age rating, runtime corrected).

- **No weighting** (decision 5, log entry 15): every film counts equally. Inverse-variance weights give the
  same result (effects correlation 1.00); count weights let 1% of films carry 71% of the audience weight.

Still to decide: (4) fan-favourite definition,
(5) weighting, (6) combining actors, (7) synergy and underminers.

Limitations to state in the app's method section:

- B1: an effect is relative to "a little-known actor/director" (someone below the film threshold).
- B2: effects are additive and constant: the same actor effect in every genre, era and cast; a genre effect
  is how liked that genre is given its cast/crew and era, not an actor's strength in that genre.
- B3: the 0-100 bounds are ignored by the linear model (ceiling: 420 films at critic score 100);
  predictions are clipped to 0-100.
- B4: the bootstrap treats films as independent; franchises/sequels are not, so ranges are slightly too narrow.
- B5: association, not causation (selection of actors into better projects).

## Data rules (strict)

- `Data/raw/`: original downloads, exactly as received. **Never edit, overwrite, rename or delete
  anything in this folder.** Every script only reads from it.
- `Data/clean/`: all cleaned, joined or derived data. It must always be reproducible from `Data/raw/` by
  running the cleaning code, so never hand-edit files here either.
- Only the student decides to add or remove raw files (the old sources were removed on their request).
- **Data in git is allowed** (decided 2026-11-16) where the app needs it on Streamlit Cloud, e.g.
  `rotten_tomatoes_movies.csv` (17 MB) and derived files in `Data/clean/`. `rotten_tomatoes_critic_reviews.csv`
  (226 MB, unused) must stay out of git: GitHub rejects files over 100 MB, so add it to `.gitignore`.

## Analysis rules

These are modelling questions with many judgement calls. Work in a short loop: **do not silently make
an assumption that can change the outcome.** Ask first, or state it explicitly at the top of the answer.
Typical decisions that need confirmation:

- How films with only one of the two scores are handled (use per score vs. require both).
- Which actors count (no billing order in the data: first N names vs. whole cast) and how name variants are
  handled.
- How films with several directors are handled.
- How the delta is defined (difference vs. ratio, sign convention), if the secondary question is done.
- Minimum number of reviews per film, and minimum number of films per actor/director, before an effect is
  estimated.
- Whether several actors' effects add up (sum) or average out (mean) in the score estimator.
- How multi-genre films are handled (all genres as separate indicators vs. one primary genre).
- Minimum number of shared films before a synergy (interaction) term is estimated.
- What counts as "clearly negative" for an underminer (effect size and uncertainty threshold).
- Which confounders to control for (genre, release year, budget, etc.).

Known pitfalls to keep in mind:

- Actors who often appear together (and with the same director) are hard to separate statistically,
  because their effects are confounded (collinear).
- Actors and directors with few films give unstable estimates; consider shrinkage/regularisation or
  minimum-count thresholds.
- Selection bias: established actors choose (and are chosen for) better projects, so association is not causation.

Always report the number of rows used and excluded, and how a result was checked.

Record accepted, adjusted or rejected AI suggestions in `beslissingslogboek.md` (via `/timtim:logboek`).
This log is the basis for the no-AI conversation at delivery.

## Environment

- Windows, conda environment `syntra-ds` (Python 3.13, channel `defaults`, not conda-forge).
  Run `conda activate syntra-ds` before running any code.
- Ask before installing packages into `syntra-ds`.
