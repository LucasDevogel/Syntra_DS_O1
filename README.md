# What drives film ratings?

**Live app:** https://syntradso1-6omrf96hgryuewtnmgv9bf.streamlit.app/

Syntra Data Scientist course, project 1. Using Rotten Tomatoes data, this project estimates how much
**actors, directors and genre** shift a film's scores, separately for **critics** (Tomatometer) and the
**audience** (Audience Score), and whether some people are clearly more liked by the public than by critics.

Both Rotten Tomatoes scores are *percentages of positive* reviews or ratings, so every effect measures
**likeability** (how many people liked a film), not film quality. An effect of +6 means 6 percentage points more
critics (or audience members) were positive than for a comparable film with a little-known actor or director.

## Research questions

1. What is the individual effect of an actor or director on the critic and the audience score, after accounting
   for genre, era and the rest of the cast and crew?
2. Who boosts scores, and who undermines them on their own?
3. How much does genre alone shift the scores?
4. Do combinations of actors (or actor and director) score better than their individual effects predict?
5. Are some people fan favourites (more liked by the audience) or critics' darlings?

## Main findings

- **People matter, mostly for critics.** Adding actors and directors lowers the error on unseen films from 19.3 to
  18.3 points (critic) and from 13.9 to 13.6 (audience). The strongest boosters are directors (e.g. Christopher
  Nolan +15.8 critic points).
- **Genre shifts scores a lot.** Documentary (+22.7 critic / +16.1 audience) and Animation (+13.3 / +12.1) score
  highest; Horror is the clearest audience underminer (−8.0).
- **Most individual effects are uncertain.** Of 3,475 people with at least 5 films, only 11% have a clear effect
  (90% range excluding 0) for both critics and audience; 26% are clear on exactly one of the two.
- **Fan favourites and critics' darlings exist:** 423 people are clearly relatively more liked by the audience,
  427 by critics.
- **Synergy could not be estimated reliably.** Only 80 pairs share at least 5 films, mostly franchises that kept
  working together after success. A profile-based "cast compatibility" effect is real but small (+1.0 critic points
  per standard deviation of cast genre similarity) and adds almost nothing to the predictions.

## How it works

| Step | What happens |
|---|---|
| Data | `rotten_tomatoes_movies.csv` (Kaggle, stefanoleone992), 17,712 films; single source, no joins |
| Filters | both scores present, release year present, >= 10 critic reviews and >= 50 audience ratings: **13,483 films** |
| People | first 5 actors (billing order) and every director; people with >= 3 films get their own effect |
| Model | one ridge regression per score: baseline + genres + decade + age rating + runtime group + cast compatibility + director(s) + sum of the actor effects. Only people are shrunk towards 0 (by about n / (n + λ)); λ = 10, chosen by 5-fold cross-validation on both scores together |
| Uncertainty | 200 bootstrap refits (resampled films), 90% percentile ranges; fan score = audience effect − 0.72 × critic effect |
| Check | prediction error on films the model never saw (cross-validation), per film shown in the app as the "honest prediction" |

Every non-trivial choice (filters, thresholds, model type, corrections, weighting, presentation) is recorded with
its reasoning in [`beslissingslogboek.md`](beslissingslogboek.md) (decision log, entries 1-24).

### Limitations

Effects are relative to a little-known actor or director; they are additive and the same in every genre and cast;
the linear model ignores the 0-100 bounds (predictions are capped); bootstrap ranges are slightly too narrow because
sequels are not independent films; and the results are associations, not causal effects (established actors
choose, and are chosen for, better projects; budget and script are not in the data).

## Repository

| Path | Contents |
|---|---|
| `pipeline.py` | data pipeline (row funnel) and the model: ridge fit, cross-validation, bootstrap, collaborations |
| `app.py` | Streamlit app (presentation only, reads `Data/clean/`) |
| `tests/` | pytest tests on hand-made data (`test_pipeline.py`, `test_model.py`) and checks on the real file (`test_real_data.py`) |
| `Data/raw/` | the original download, never edited |
| `Data/clean/` | everything derived from `Data/raw/` by `pipeline.py`, committed so the deployed app can read it |
| `beslissingslogboek.md` | decision log |
| `CLAUDE.md` | project instructions and working notes |

The critic reviews file of the same Kaggle dataset (226 MB) is not used and not in the repository.

## Running it locally

```powershell
conda activate syntra-ds
python pipeline.py          # rebuilds Data/clean/ (about 5 minutes)
python -m pytest            # runs the tests
streamlit run app.py        # opens the app at http://localhost:8501
```

The deployed app only needs the packages in `requirements.txt` (Python 3.13).
