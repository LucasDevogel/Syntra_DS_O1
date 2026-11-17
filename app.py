"""Streamlit app: what drives Rotten Tomatoes critic and audience scores?

Reads the files that `python pipeline.py` writes to Data/clean/ (films, funnel, model coefficients, bootstrap
draws, actor profiles, collaborations). All modelling lives in pipeline.py; this file only presents it.
Run locally with: streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as p

st.set_page_config(page_title="What drives film ratings?", page_icon="🎬", layout="wide")

# Colours: slots 1-3 of the dataviz reference palette (validated colour-blind safe as a set, light and dark).
# Throughout the app orange = critics, aqua = audience, blue = both.
PALETTE = {
    "light": {"critic": "#eb6834", "audience": "#1baf7a", "both": "#2a78d6", "line": "#52514e", "text": "#0b0b0b"},
    "dark": {"critic": "#d95926", "audience": "#199e70", "both": "#3987e5", "line": "#c3c2b7", "text": "#f0efec"},
}
CLARITY = ["Both effects clear", "Only critic effect clear", "Only audience effect clear", "Neither clear (faded)"]
COLORS = PALETTE["dark" if st.context.theme.type == "dark" else "light"]
QUADRANTS = {  # (critic positive, audience positive) -> name (entry 22)
    (True, True): "All-round star",
    (False, True): "Fan favourite",
    (True, False): "Critics' darling",
    (False, False): "Underperformer",
}
RANKING_MIN_FILMS = 5  # entry 20: lists and rankings need >= 5 films


@st.cache_resource
def load_data():
    spec, coefs, draws, info = p.load_model()
    films = pd.read_csv(p.CLEAN_DIR / "films.csv", keep_default_na=False, na_values={"runtime": [""], "content_rating": [""]})
    roles = pd.read_csv(p.CLEAN_DIR / "film_roles.csv", keep_default_na=False)
    funnel = pd.read_csv(p.CLEAN_DIR / "funnel.csv", keep_default_na=False)
    cv = pd.read_csv(p.CLEAN_DIR / "model_cv.csv")
    collabs = pd.read_csv(p.CLEAN_DIR / "collaborations.csv", keep_default_na=False)
    honest = pd.read_csv(p.CLEAN_DIR / "film_predictions.csv").set_index("film_id")
    lists = roles.groupby(["film_id", "role"], sort=False)["name"].agg(list).unstack("role")
    films = films.join(lists.rename(columns={"actor": "actors", "director": "directors", "genre": "genres"}), on="film_id")
    for col in ["actors", "directors", "genres"]:
        films[col] = films[col].map(lambda v: v if isinstance(v, list) else [])
    films = films.join(honest, on="film_id")
    people = coefs[coefs["kind"].isin(p.PERSON_KINDS)].copy()
    people["quadrant"] = [QUADRANTS[(c >= 0, a >= 0)] for c, a in zip(people["critic_effect"], people["audience_effect"])]
    excl = lambda s: (people[f"{s}_low"] > 0) | (people[f"{s}_high"] < 0)  # noqa: E731
    people["critic_clear"], people["audience_clear"] = excl("critic"), excl("audience")
    people["both_clear"] = people["critic_clear"] & people["audience_clear"]
    people["fan_clear"] = (people["fan_low"] > 0) | (people["fan_high"] < 0)
    return spec, coefs, draws, info, films, roles, funnel, cv, collabs, people


spec, coefs, draws, info, films, roles, funnel, cv, collabs, people = load_data()
est = coefs[["critic_effect", "audience_effect"]].to_numpy()


def signed(x: float) -> str:
    return f"{x:+.1f}"


def with_range(effect: float, low: float, high: float) -> str:
    return f"{effect:+.1f} (90% range {low:+.1f} to {high:+.1f})"


def films_of(name: str, role: str) -> pd.DataFrame:
    ids = roles.loc[(roles["role"] == role) & (roles["name"] == name), "film_id"]
    out = films[films["film_id"].isin(ids)][["title", "release_year", "critic_score", "audience_score"]]
    return out.sort_values("release_year").rename(columns={"title": "Film", "release_year": "Year",
                                                           "critic_score": "Critic score", "audience_score": "Audience score"})


def match_name(typed: str, role: str) -> str | None:
    """Exact name as it appears in the data (case-insensitive match), or None."""
    names = roles.loc[roles["role"] == role, "name"]
    hits = names[names.str.lower() == typed.strip().lower()]
    return hits.iloc[0] if len(hits) else None


cv_best = cv[cv["lambda"] == info["lambda"]].groupby(["variant", "score"])["mae"].mean().unstack("score")


def estimate(film: pd.DataFrame, exclude_self: bool):
    """Model estimate for one film: (raw sum, capped estimate, 90% range per score, breakdown, little-known names).

    exclude_self=True for a film that is in the data (its own genres/year are left out of the cast profiles, as
    in training); False for a new combination.
    """
    xrow = p.build_design(film, spec, exclude_self).toarray()[0]
    raw = xrow @ est
    boot = np.clip(np.tensordot(draws, xrow, axes=([1], [0])), 0, 100)  # one prediction per bootstrap model
    low, high = np.percentile(boot, info["range_percentiles"], axis=0)
    kinds, names = spec.columns["kind"].to_numpy(), spec.columns["name"].to_numpy()
    contrib = xrow[:, None] * est
    row = film.iloc[0]
    parts = [("Baseline (decade, age rating, runtime)",
              contrib[np.isin(kinds, ["baseline", "decade", "age_rating", "runtime"])].sum(0), "")]
    for gname in row["genres"]:
        parts.append((f"Genre: {gname}", contrib[(kinds == "genre") & (names == gname)].sum(0), ""))
    little_known = []
    for kind, who in [("director", row["directors"]), ("actor", row["actors"])]:
        for person in who:
            mask = (kinds == kind) & (names == person)
            if mask.any():
                n = int(spec.columns.loc[mask, "n_films"].iloc[0])
                parts.append((f"{kind.capitalize()}: {person}", contrib[mask].sum(0), f"{n} films"))
            else:
                little_known.append(person)
                parts.append((f"{kind.capitalize()}: {person}", np.zeros(2), f"little-known (< {p.MIN_FILMS_PERSON} films): 0"))
    parts.append(("Cast compatibility", contrib[kinds == "compatibility"].sum(0), "needs actors with >= 2 other films; else 0"))
    breakdown = pd.DataFrame([{"Part": n, "Critic": v[0], "Audience": v[1], "Note": note} for n, v, note in parts])
    return raw, np.clip(raw, 0, 100), low, high, breakdown, little_known


@st.cache_data
def miss_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Honest-prediction error per band of actual scores, and how often actual vs. predicted scores are extreme."""
    bands, extremes = {}, {}
    for s, label in [("critic", "Critic"), ("audience", "Audience")]:
        actual, pred = films[f"{s}_score"], films[f"{s}_honest"]
        band = pd.cut(actual, [-1, 20, 40, 60, 80, 100], labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
        err = (actual - pred).groupby(band, observed=True)
        bands[(label, "Films")] = err.size()
        bands[(label, "Mean actual − predicted")] = err.mean()
        bands[(label, "Mean abs. error")] = (actual - pred).abs().groupby(band, observed=True).mean()
        for name, a, b in [("actual", actual, None), ("predicted", pred, None)]:
            extremes[f"{label} {name}"] = {
                "Middle 90% of films": f"{a.quantile(0.05):.0f}-{a.quantile(0.95):.0f}",
                "Score <= 10": f"{(a <= 10).mean():.1%}",
                "Score >= 90": f"{(a >= 90).mean():.1%}",
                "Lowest / highest": f"{a.min():.0f} / {a.max():.0f}",
            }
    out = pd.DataFrame(bands).round(1)
    out.columns = [f"{a}: {b}" for a, b in out.columns]
    return out.rename_axis("Actual score"), pd.DataFrame(extremes)


def show_breakdown(breakdown: pd.DataFrame, raw: np.ndarray, container=st) -> None:
    container.dataframe(breakdown, hide_index=True, width="stretch",
                        column_config={"Critic": st.column_config.NumberColumn(format="%+.1f"),
                                       "Audience": st.column_config.NumberColumn(format="%+.1f")})
    container.caption(f"The parts add up to {raw[0]:.1f} (critic) and {raw[1]:.1f} (audience) before capping at 0-100.")

# --------------------------------------------------------------------------------------------------------
st.title("What drives film ratings?")
st.markdown(
    "How much do **actors, directors and genre** shift a film's **Rotten Tomatoes** scores, separately for "
    "**critics** (Tomatometer) and the **audience** (Audience Score)? And are some people clearly more liked "
    "by the public than by critics?"
)
tabs = st.tabs(["Method", "Look up a person", "Effect matrix", "Rankings", "Genres", "Score estimator",
                "Check a film", "Recurring collaborations"])

# --------------------------------------------------------------------------------------------------------
with tabs[0]:
    st.header("Research question")
    st.markdown(
        """
1. What is the **individual effect** of an actor or a director on the critic score and on the audience score,
   after accounting for genre, era and the rest of the cast and crew?
2. Who **boosts** scores, and who **undermines** them on their own?
3. How much does **genre** alone shift the scores?
4. Do **combinations** of actors work better or worse than their individual effects predict?
5. Are some people **fan favourites** (liked more by the audience than by critics) or **critics' darlings**?
"""
    )
    st.info(
        "**How to read an effect.** Both Rotten Tomatoes scores are *percentages of positive* reviews or ratings, "
        "not average grades. An effect of **+6** means *6 percentage points more critics (or audience members) "
        "were positive* than for a comparable film with a little-known actor or director in that role. "
        "It measures **likeability**, not film quality."
    )

    st.header("From raw data to result")
    st.subheader("1. Data and row funnel")
    st.markdown(
        "Single source: the *Rotten Tomatoes movies and critic reviews* dataset (Kaggle, stefanoleone992), "
        f"file `rotten_tomatoes_movies.csv` with {int(funnel['rows_in'].iloc[0]):,} films. "
        "No other datasets are joined."
    )
    st.dataframe(
        funnel.rename(columns={"step": "Step", "rows_in": "Rows in", "rows_out": "Rows out", "dropped": "Dropped",
                               "reason": "Why dropped"}),
        hide_index=True, width="stretch",
    )
    st.markdown(
        f"""
- **Actors:** only the first {p.TOP_N_ACTORS} names in the cast list (billing order); a name listed twice counts once.
- **Films without actors or director are kept:** they still inform the genre and era effects.
- **Every genre and every director** listed for a film counts for that film.
"""
    )

    st.subheader("2. The model")
    st.markdown(
        """
One **ridge regression** per score, on the same films and columns:

> score = baseline + genres + decade + age rating + runtime + cast compatibility + director(s) + sum of the top-5 actor effects

- **Why ridge, not ordinary regression (OLS):** people with few films get extreme OLS estimates (noise is credited
  to them) and actors who always appear together cannot be separated. Ridge **shrinks** each person's effect
  towards 0 by roughly *n / (n + λ)* for someone with *n* films: little evidence means a cautious estimate.
- **Only people are shrunk.** Genre, decade, age rating and runtime are control variables, so their effects are
  not absorbed by the people (and vice versa). Decade matters most: older films on Rotten Tomatoes are mostly
  classics, so without it classic actors would get credit for their era.
- **Minimum films:** actors and directors with at least 3 films get their own effect; everyone else counts as
  *little-known* (effect 0). Rankings only show people with at least 5 films.
- **Cast compatibility:** three features from the actors' *other* films (genres and years only, never scores):
  does the film's genre match what the cast usually plays, do the actors play similar genres, are they from
  the same era.
- **Uncertainty:** the model is refitted on 200 resamples of the films (bootstrap). The **90% range** is the
  middle 90% of those estimates.
"""
    )
    st.markdown(
        f"**Check on unseen films** (5-fold cross-validation, λ = {info['lambda']:g} chosen on both scores together): "
        "mean absolute error in score points."
    )
    st.dataframe(
        cv_best.loc[list(p.CV_VARIANTS)].rename(columns={"critic_score": "Critic score", "audience_score": "Audience score"})
        .rename_axis("Model").round(2),
        width="content",
    )
    st.caption(
        "Actors and directors clearly improve predictions on films the model has not seen. Cast compatibility "
        "adds very little once it is estimated together with the people; it is kept because it costs little and "
        "lets the score estimator judge new casts."
    )

    st.subheader("3. Why synergy could not be estimated reliably")
    st.markdown(
        """
Three approaches were tried: a separate effect per pair, cast *profiles* (genre mix and era, so new combinations can
be judged), and actors' past performance per genre. None gives a reliable synergy estimate with this data:

- **Too few repeats:** only 80 pairs share at least 5 films, while a user almost always picks a combination that never
  existed. A single film deviates about 22 critic points from the prediction, so the average of 5 shared films is still
  uncertain by roughly 10 points; measuring a synergy of a few points would take about 100 shared films.
- **Biased selection:** pairs only keep working together after success (almost no negative values), and many are one
  franchise, so "synergy" mixes chemistry with the success of those specific films.
- **The data lacks what drives chemistry:** profiles describe *what* actors play, not *how* they play or get along
  (actors with similar profiles do not have more similar effects). The small profile effect that was found is mostly
  absorbed by the individual effects (gain on unseen films: 0.02 points). Using past performance was leakage-prone
  and gave contradictory signs.

So the conclusion is *no evidence with this data*, not *synergy does not exist*. The app keeps the profile features as
**cast compatibility** (a small, real effect) and shows the measurable pairs only as an exploratory list.
"""
    )

    st.subheader("4. Limitations")
    st.markdown(
        """
- **Relative effects:** an effect is relative to *a little-known actor or director* (fewer than 3 films).
- **Additive and constant:** an actor has the same effect in every genre, era and cast. A genre effect is how
  liked that genre is given its cast and era, not an actor's strength in that genre.
- **0-100 bounds:** the linear model ignores them (420 films have a critic score of exactly 100); predictions are
  capped at 0-100.
- **Ranges slightly too narrow:** the bootstrap treats films as independent; sequels and franchises are not.
- **Association, not causation:** established actors choose, and are chosen for, better projects. Budget, script
  and studio are not in the data.
"""
    )

# --------------------------------------------------------------------------------------------------------
with tabs[1]:
    st.header("Look up an actor or director")
    role = st.radio("Role", ["actor", "director"], horizontal=True, format_func=str.capitalize, key="lookup_role")
    options = sorted(people.loc[people["kind"] == role, "name"])
    typed = st.selectbox(f"Name ({len(options):,} {role}s with an own effect; you can also type any other name)",
                         options, index=None, placeholder="Start typing a name...", accept_new_options=True)
    if typed:
        row = people[(people["kind"] == role) & (people["name"] == typed)]
        if row.empty:
            name = match_name(typed, role)
            if name is None:
                st.warning(f"No {role} called '{typed}' in the {len(films):,} films. Check the spelling.")
            else:
                n = int(roles[(roles["role"] == role) & (roles["name"] == name)]["film_id"].nunique())
                st.info(f"**{name}** appears in {n} film(s), fewer than {p.MIN_FILMS_PERSON}. The model treats them as a "
                        f"*little-known* {role}: **effect 0** on both scores, because there is too little evidence.")
                st.dataframe(films_of(name, role), hide_index=True)
        else:
            r = row.iloc[0]
            st.subheader(r["name"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Critic effect", signed(r["critic_effect"]), border=True)
            c1.caption(f"90% range {r['critic_low']:+.1f} to {r['critic_high']:+.1f}")
            c2.metric("Audience effect", signed(r["audience_effect"]), border=True)
            c2.caption(f"90% range {r['audience_low']:+.1f} to {r['audience_high']:+.1f}")
            c3.metric("Fan score", signed(r["fan_score"]), border=True)
            c3.caption(f"90% range {r['fan_low']:+.1f} to {r['fan_high']:+.1f}")
            if r["fan_low"] > 0:
                fan_text = "clearly **more liked by the audience** than by critics (relative to how both usually score)"
            elif r["fan_high"] < 0:
                fan_text = "clearly **more liked by critics** than by the audience (relative to how both usually score)"
            else:
                fan_text = "**no clear difference** between audience and critics"
            certainty = ("both ranges exclude 0, so this placement is clear" if r["both_clear"]
                         else "at least one range includes 0, so this placement is uncertain")
            st.markdown(
                f"Based on **{int(r['n_films'])} films**. Quadrant: **{r['quadrant']}** ({certainty}). Fan score: {fan_text}."
            )
            if r["n_films"] < RANKING_MIN_FILMS:
                st.caption(f"Fewer than {RANKING_MIN_FILMS} films: shown here, but left out of the rankings and the matrix by default.")
            st.caption("Fan score = audience effect − 0.72 × critic effect. Critic scores spread more (sd 27.7 vs. 20.0), "
                       "so the critic effect is first converted to audience points.")
            st.dataframe(films_of(r["name"], role), hide_index=True)

# --------------------------------------------------------------------------------------------------------
with tabs[2]:
    st.header("Effect matrix: critics vs. audience")
    c1, c2, c3 = st.columns([1, 2, 2])
    role = c1.radio("Show", ["actor", "director"], horizontal=True, format_func=lambda r: r.capitalize() + "s",
                    key="matrix_role")
    min_films = c2.slider("Minimum number of films", p.MIN_FILMS_PERSON, 30, RANKING_MIN_FILMS)
    groups = c3.multiselect("Groups to show", CLARITY, default=CLARITY)
    shown = people[(people["kind"] == role) & (people["n_films"] >= min_films)].copy()
    shown["clarity"] = np.select([shown["both_clear"], shown["critic_clear"], shown["audience_clear"]], CLARITY[:3], CLARITY[3])
    all_shown = shown  # before the group filter, for the counts below
    shown = shown[shown["clarity"].isin(groups)]
    highlight = st.multiselect("Highlight people (they stay visible whatever the group filter)", sorted(all_shown["name"]),
                               placeholder="Optional: type one or more names")
    shown = shown.sort_values("clarity", key=lambda s: s.eq(CLARITY[3]), ascending=False)  # faded points drawn first
    for df in (shown, all_shown):
        for s in ["critic", "audience"]:
            df[f"{s}_text"] = [with_range(e, lo, hi) for e, lo, hi in zip(df[f"{s}_effect"], df[f"{s}_low"], df[f"{s}_high"])]

    # Axis limits from all people of this role and film count, so the scale does not jump when filtering groups
    lim_x = float(np.ceil(all_shown["critic_effect"].abs().max() + 1)) if len(all_shown) else 10
    lim_y = float(np.ceil(all_shown["audience_effect"].abs().max() + 1)) if len(all_shown) else 10
    x = alt.X("critic_effect:Q", title="Critic effect (points)", scale=alt.Scale(domain=[-lim_x, lim_x]))
    y = alt.Y("audience_effect:Q", title="Audience effect (points)", scale=alt.Scale(domain=[-lim_y, lim_y]))
    points = alt.Chart(shown).mark_circle(size=60, stroke="white", strokeWidth=0.5).encode(
        x=x, y=y,
        color=alt.Color("clarity:N", title=None, sort=CLARITY,
                        scale=alt.Scale(domain=CLARITY, range=[COLORS["both"], COLORS["critic"], COLORS["audience"], COLORS["line"]]),
                        legend=alt.Legend(orient="bottom", columns=2)),
        opacity=alt.Opacity("clarity:N", scale=alt.Scale(domain=CLARITY, range=[0.9, 0.9, 0.9, 0.2]), legend=None),
        tooltip=[alt.Tooltip("name:N", title="Name"), alt.Tooltip("n_films:Q", title="Films"),
                 alt.Tooltip("quadrant:N", title="Quadrant"), alt.Tooltip("clarity:N", title="Clear (90% range excludes 0)"),
                 alt.Tooltip("critic_text:N", title="Critic effect"),
                 alt.Tooltip("audience_text:N", title="Audience effect"),
                 alt.Tooltip("fan_score:Q", title="Fan score", format="+.1f")],
    )
    axes = alt.Chart(pd.DataFrame({"zero": [0]}))
    zero_lines = axes.mark_rule(color=COLORS["line"], strokeWidth=1).encode(x="zero:Q") + \
        axes.mark_rule(color=COLORS["line"], strokeWidth=1).encode(y="zero:Q")
    diag = alt.Chart(pd.DataFrame({"critic_effect": [-lim_x, lim_x], "audience_effect": [-p.FAN_FACTOR * lim_x, p.FAN_FACTOR * lim_x]})) \
        .mark_line(color=COLORS["line"], strokeDash=[6, 4], strokeWidth=1.5, clip=True).encode(x=x, y=y)
    labels = pd.DataFrame({
        "critic_effect": [0.97 * lim_x, -0.97 * lim_x, 0.97 * lim_x, -0.97 * lim_x],
        "audience_effect": [0.97 * lim_y, 0.97 * lim_y, -0.97 * lim_y, -0.97 * lim_y],
        "label": ["All-round star", "Fan favourite", "Critics' darling", "Underperformer"],
        "align": ["right", "left", "right", "left"],
        "baseline": ["top", "top", "bottom", "bottom"],
    })
    corner = [alt.Chart(labels.iloc[[i]]).mark_text(fontSize=13, fontWeight="bold", color=COLORS["line"],
                                                    align=labels["align"][i], baseline=labels["baseline"][i])
              .encode(x=x, y=y, text="label:N") for i in range(4)]
    layers = [zero_lines, diag, points, *corner]
    if highlight:
        h = all_shown[all_shown["name"].isin(highlight)]
        tips = [alt.Tooltip("name:N", title="Name"), alt.Tooltip("n_films:Q", title="Films"),
                alt.Tooltip("clarity:N", title="Clear"), alt.Tooltip("critic_text:N", title="Critic effect"),
                alt.Tooltip("audience_text:N", title="Audience effect")]
        layers.append(alt.Chart(h).mark_circle(size=60).encode(
            x=x, y=y, tooltip=tips,
            color=alt.Color("clarity:N", scale=alt.Scale(domain=CLARITY, range=[COLORS["both"], COLORS["critic"], COLORS["audience"], COLORS["line"]]), legend=None)))
        layers.append(alt.Chart(h).mark_circle(size=220, filled=False, stroke=COLORS["text"], strokeWidth=2.5).encode(x=x, y=y, tooltip=tips))
        layers.append(alt.Chart(h).mark_text(dx=12, align="left", fontWeight="bold", color=COLORS["text"]).encode(x=x, y=y, text="name:N"))
    st.altair_chart(alt.layer(*layers).properties(height=600), width="stretch")
    if len(groups) < len(CLARITY):
        st.caption(f"Showing {len(shown):,} of {len(all_shown):,} {role}s (group filter active).")

    n = max(len(all_shown), 1)
    k = all_shown["clarity"].value_counts().reindex(CLARITY, fill_value=0)
    st.markdown(
        f"**{len(all_shown):,} {role}s** with at least {min_films} films. A range is *clear* when the 90% range lies entirely "
        "on one side of 0.\n"
        f"- **Blue:** both effects clear, {k[CLARITY[0]]:,} ({k[CLARITY[0]] / n:.0%}): the quadrant is firm.\n"
        f"- **Orange:** only the critic effect clear, {k[CLARITY[1]]:,} ({k[CLARITY[1]] / n:.0%}): left/right is firm, up/down is not.\n"
        f"- **Aqua:** only the audience effect clear, {k[CLARITY[2]]:,} ({k[CLARITY[2]] / n:.0%}): up/down is firm, left/right is not.\n"
        f"- **Faded grey:** neither clear, {k[CLARITY[3]]:,} ({k[CLARITY[3]] / n:.0%}): they only *lean* towards their quadrant; "
        "a slightly different set of films could move them."
    )
    st.markdown(
        "The **dashed line** is audience effect = 0.72 × critic effect. Points **above** it are relatively more liked by "
        "the audience than by critics (positive fan score), points **below** it relatively more by critics."
    )
    counts = pd.crosstab(all_shown["quadrant"], all_shown["clarity"]).reindex(index=list(QUADRANTS.values()), columns=CLARITY, fill_value=0)
    st.dataframe(counts.rename_axis(index="Quadrant", columns=None), width="content")
    with st.expander("Table view"):
        st.dataframe(shown[["name", "n_films", "quadrant", "critic_effect", "critic_low", "critic_high", "audience_effect",
                            "audience_low", "audience_high", "fan_score"]].sort_values("name"),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%+.1f") for c in
                                    ["critic_effect", "critic_low", "critic_high", "audience_effect", "audience_low",
                                     "audience_high", "fan_score"]})

# --------------------------------------------------------------------------------------------------------
with tabs[3]:
    st.header("Rankings")
    st.markdown(f"Only people with at least **{RANKING_MIN_FILMS} films**. A tick in *Clear* means the 90% range lies "
                "entirely on one side of 0.")
    c1, c2, c3 = st.columns(3)
    measure = c1.selectbox("Measure", ["critic", "audience", "fan"], format_func={
        "critic": "Critic effect", "audience": "Audience effect", "fan": "Fan score (audience vs. critics)"}.get)
    who = c2.radio("Who", ["both", "actor", "director"], horizontal=True,
                   format_func={"both": "Both", "actor": "Actors", "director": "Directors"}.get)
    top_n = c3.slider("How many", 5, 50, 15)
    pool = people[people["n_films"] >= RANKING_MIN_FILMS]
    if who != "both":
        pool = pool[pool["kind"] == who]
    col = "fan_score" if measure == "fan" else f"{measure}_effect"
    lo, hi = ("fan_low", "fan_high") if measure == "fan" else (f"{measure}_low", f"{measure}_high")
    clear = "fan_clear" if measure == "fan" else f"{measure}_clear"
    view = pool[["name", "kind", "n_films", col, lo, hi, clear]].rename(columns={
        "name": "Name", "kind": "Role", "n_films": "Films", col: "Effect", lo: "Low (5%)", hi: "High (95%)", clear: "Clear"})
    fmt = {c: st.column_config.NumberColumn(format="%+.1f") for c in ["Effect", "Low (5%)", "High (95%)"]}
    top_title, bottom_title = (("Fan favourites", "Critics' favourites") if measure == "fan" else ("Boosters", "Underminers"))
    left, right = st.columns(2)
    left.subheader(top_title)
    left.dataframe(view.nlargest(top_n, "Effect"), hide_index=True, column_config=fmt, width="stretch")
    right.subheader(bottom_title)
    right.dataframe(view.nsmallest(top_n, "Effect"), hide_index=True, column_config=fmt, width="stretch")
    n_pos, n_neg = int((pool[lo] > 0).sum()), int((pool[hi] < 0).sum())
    st.caption(f"Of {len(pool):,} people shown, {n_pos:,} have a range entirely above 0 and {n_neg:,} entirely below 0 "
               f"(by chance alone about {0.05 * len(pool):,.0f} each would be expected).")

# --------------------------------------------------------------------------------------------------------
with tabs[4]:
    st.header("Genre effects")
    g = coefs[coefs["kind"] == "genre"].copy()
    g["label"] = [f"{n} ({int(k):,} films)" for n, k in zip(g["name"], g["n_films"])]
    order = list(g.sort_values("critic_effect", ascending=False)["label"])
    long = pd.concat([
        g.assign(score=s.capitalize(), effect=g[f"{s}_effect"], low=g[f"{s}_low"], high=g[f"{s}_high"])
        for s in ["critic", "audience"]
    ])
    long["text"] = [with_range(e, a, b) for e, a, b in zip(long["effect"], long["low"], long["high"])]
    color = alt.Color("score:N", title=None, scale=alt.Scale(domain=["Critic", "Audience"],
                                                             range=[COLORS["critic"], COLORS["audience"]]),
                      legend=alt.Legend(orient="top"))
    base = alt.Chart(long).encode(y=alt.Y("label:N", sort=order, title=None), yOffset=alt.YOffset("score:N", sort=["Critic", "Audience"]),
                                  color=color,
                                  tooltip=[alt.Tooltip("name:N", title="Genre"), alt.Tooltip("score:N", title="Score"),
                                           alt.Tooltip("text:N", title="Effect"), alt.Tooltip("n_films:Q", title="Films")])
    ranges = base.mark_rule(strokeWidth=2).encode(x=alt.X("low:Q", title="Effect on the score (points)"), x2="high:Q")
    dots = base.mark_circle(size=70, opacity=1).encode(x="effect:Q")
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=COLORS["line"]).encode(x="zero:Q")
    st.altair_chart(alt.layer(zero, ranges, dots).properties(height=alt.Step(14)), width="stretch")
    st.markdown(
        "An effect is the shift for a film **with** this genre tag compared with one without it, for the same cast, "
        "director, decade, age rating and runtime. Films often have several genres; each tag adds its own effect. "
        "Lines show the 90% range; small genres (e.g. Anime & Manga, 14 films) have wide ranges."
    )
    with st.expander("Table view, and the other corrections (decade, age rating, runtime)"):
        for kind, title in [("genre", "Genre"), ("decade", "Decade"), ("age_rating", "Age rating"), ("runtime", "Runtime (minutes)")]:
            part = coefs[coefs["kind"] == kind][["name", "n_films", "critic_effect", "critic_low", "critic_high",
                                                  "audience_effect", "audience_low", "audience_high"]]
            st.markdown(f"**{title}**")
            st.dataframe(part.rename(columns={"name": title, "n_films": "Films"}), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%+.1f") for c in part.columns[2:]})
        st.caption("Decade, age rating and runtime are groups of which every film has exactly one, so their levels "
                   "only mean something *relative to each other* (e.g. 1950s vs. 2010s), not on their own.")

# --------------------------------------------------------------------------------------------------------
with tabs[5]:
    st.header("Score estimator")
    st.markdown("Pick a director, up to 5 actors and one or more genres. The estimate uses the same model as the effects.")
    directors = sorted(people.loc[people["kind"] == "director", "name"])
    actors = sorted(people.loc[people["kind"] == "actor", "name"])
    genres = sorted(coefs.loc[coefs["kind"] == "genre", "name"])
    c1, c2 = st.columns(2)
    director = c1.selectbox("Director", directors, index=None, placeholder="Optional; type any name",
                            accept_new_options=True)
    cast = c1.multiselect("Actors (max 5, order does not matter)", actors, max_selections=5,
                          placeholder="Type names", accept_new_options=True)
    chosen_genres = c1.multiselect("Genre(s)", genres, default=["Drama"])
    decades = [d for d in coefs.loc[coefs["kind"] == "decade", "name"]]
    decades = sorted(decades, key=lambda d: (0, d) if d.startswith("before") else (1, d))
    decade = c2.selectbox("Decade", decades, index=decades.index("2010s"))
    ratings = list(coefs.loc[coefs["kind"] == "age_rating", "name"])
    rating = c2.selectbox("Age rating", ratings, index=ratings.index("R"))
    minutes = c2.slider("Runtime (minutes)", 60, 200, 100)
    group = p.add_categories(pd.DataFrame({"release_year": [2000], "content_rating": ["R"], "runtime": [float(minutes)]}))["runtime_group"].iloc[0]
    c2.caption(f"The model uses runtime groups: {minutes} minutes falls in **{group}**, so the estimate changes in steps.")

    year = p.FIRST_DECADE - 5 if decade.startswith("before") else int(decade[:4]) + 5
    film = pd.DataFrame([{"release_year": year, "content_rating": rating, "runtime": float(minutes),
                          "genres": chosen_genres, "directors": [director] if director else [], "actors": cast}])
    raw, pred, low, high, breakdown, little_known = estimate(film, exclude_self=False)
    m1, m2 = c2.columns(2)
    for k, (box, label) in enumerate([(m1, "Critic score"), (m2, "Audience score")]):
        box.metric(label, f"{pred[k]:.0f}", border=True)
        box.caption(f"90% range {low[k]:.0f}-{high[k]:.0f}")
    if not chosen_genres:
        st.warning("No genre selected: the estimate then has no genre effect at all.")
    if (raw < 0).any() or (raw > 100).any():
        st.warning("The linear model went beyond the 0-100 scale; the estimate is capped at 0-100.")
    mae = cv_best.loc["full model"]
    st.caption(
        f"The range shows how certain the model is about the **average** film with this setup. A single real film "
        f"deviates much more: on unseen films the model is off by {mae['critic_score']:.0f} (critic) and "
        f"{mae['audience_score']:.0f} (audience) points on average."
    )

    st.subheader("How the estimate is built")
    show_breakdown(breakdown, raw)
    if little_known:
        st.info(f"**{', '.join(little_known)}**: fewer than {p.MIN_FILMS_PERSON} films in the data (or not in it at all), "
                "so they count as a little-known person with effect 0. The estimate is not hiding them: there is simply "
                "too little evidence for an own effect.")

# --------------------------------------------------------------------------------------------------------
with tabs[6]:
    st.header("Check a film: actual scores vs. the model")
    st.markdown(f"Pick any of the {len(films):,} films in the data and compare its real Rotten Tomatoes scores with the model.")
    labels = (films["title"] + " (" + films["release_year"].astype(str) + ")").tolist()
    pick = st.selectbox("Film", range(len(films)), index=None, format_func=lambda i: labels[i],
                        placeholder="Start typing a title...")
    if pick is not None:
        f = films.iloc[pick]
        film = films.iloc[[pick]][["release_year", "content_rating", "runtime", "genres", "directors", "actors"]]
        raw, pred, low, high, breakdown, little_known = estimate(film, exclude_self=True)
        left, right = st.columns(2, gap="large")

        left.subheader("Actual")
        a1, a2 = left.columns(2)
        a1.metric("Critic score", f"{f['critic_score']:.0f}", border=True)
        a1.caption(f"{int(f['critic_count']):,} critic reviews")
        a2.metric("Audience score", f"{f['audience_score']:.0f}", border=True)
        a2.caption(f"{int(f['audience_count']):,} audience ratings")
        runtime = "unknown" if pd.isna(f["runtime"]) else f"{f['runtime']:.0f} min"
        details = pd.DataFrame({"": ["Year", "Genres", "Director(s)", f"Top-{p.TOP_N_ACTORS} actors", "Age rating", "Runtime"],
                                "Value": [str(f["release_year"]), ", ".join(f["genres"]) or "-", ", ".join(f["directors"]) or "-",
                                          ", ".join(f["actors"]) or "-", str(f["content_rating"]), runtime]})
        left.dataframe(details, hide_index=True, width="stretch")

        right.subheader("Model")
        r1, r2 = right.columns(2)
        for k, (box, label, actual) in enumerate([(r1, "Critic score", f["critic_score"]), (r2, "Audience score", f["audience_score"])]):
            box.metric(f"{label}: model fit", f"{pred[k]:.0f}", delta=f"{pred[k] - actual:+.0f} vs. actual",
                       delta_color="off", border=True)
            box.caption(f"90% range {low[k]:.0f}-{high[k]:.0f}")
        h1, h2 = right.columns(2)
        for box, label, value, actual in [(h1, "Critic score", f["critic_honest"], f["critic_score"]),
                                          (h2, "Audience score", f["audience_honest"], f["audience_score"])]:
            box.metric(f"{label}: honest prediction", f"{value:.0f}", delta=f"{value - actual:+.0f} vs. actual",
                       delta_color="off", border=True)
        right.caption(
            "**Model fit** = the final model, which was trained on all films *including this one*, broken down below. "
            "**Honest prediction** = a model trained without this film (cross-validation), so a fair test of how well the "
            "model predicts a film it does not know. The gap between the two shows how much the model 'remembers' known films."
        )
        show_breakdown(breakdown, raw, right)
        if little_known:
            right.info(f"Little-known (fewer than {p.MIN_FILMS_PERSON} films, effect 0): {', '.join(little_known)}.")
        mae = cv_best.loc["full model"]
        st.caption(f"For reference: across all films, the honest prediction is off by {mae['critic_score']:.1f} (critic) and "
                   f"{mae['audience_score']:.1f} (audience) points on average.")

    with st.expander("Why are big hits and flops missed by so much?", expanded=pick is not None):
        st.markdown(
            "The model knows a film's **starting conditions** (genre, era, age rating, runtime and who made it), not **how the "
            "film turned out** (script, execution, budget, marketing, word of mouth). So it estimates the *typical* reception "
            "of a film with this setup. Films that became extreme hits or flops are usually extreme partly for reasons nobody "
            "could see in advance, so their predictions lie **closer to the middle** (*regression to the mean*; ridge's "
            "shrinkage adds a little to this on purpose)."
        )
        bands, extremes = miss_tables()
        st.markdown("**Honest predictions for all films, grouped by the actual score.** Negative = predicted too high.")
        st.dataframe(bands, width="content", column_config={c: st.column_config.NumberColumn(format="%+.1f")
                                                            for c in bands.columns if "actual" in c})
        st.markdown("**How often scores are extreme: actual vs. predicted.**")
        st.dataframe(extremes, width="content")
        st.caption("Very low scores are never predicted and very high ones rarely (a prediction reaches 100 only through "
                   "the 0-100 cap). Some effects push strongly upwards (e.g. Documentary, Animation, top directors), while "
                   "hardly anything pushes down as hard.")

# --------------------------------------------------------------------------------------------------------
with tabs[7]:
    st.header("Standout recurring collaborations (exploratory)")
    st.markdown(
        f"Pairs with at least {p.MIN_SHARED_FILMS} shared films, where each partner also has at least "
        f"{p.MIN_FILMS_APART} films without the other (no actor-directors directing themselves). *Surprise* = how much "
        "their shared films scored above (or below) what the model predicts from the individual effects, with each "
        "film predicted by a model that never saw it. Groups that always appear together are shown as one row."
    )
    st.warning(
        "**Read with caution, this is not proof of chemistry.**\n"
        "- **Franchise effect:** many rows are one series (Toy Story, Harry Potter, The Godfather). The surprise is "
        "about *those films*, whatever the reason.\n"
        "- **Selection:** pairs only keep working together after success, so negative values are almost absent.\n"
        "- **Ranges too narrow:** they ignore the model's own uncertainty, and series films are not independent.\n"
        "- **Not usable for new combinations:** the score estimator uses cast compatibility instead."
    )
    view = collabs.copy()
    for s in ["critic", "audience"]:
        view[f"{s.capitalize()} surprise"] = [with_range(e, a, b) for e, a, b in zip(view[f"{s}_surprise"], view[f"{s}_low"], view[f"{s}_high"])]
    sort = st.radio("Sort by", ["Critic", "Audience"], horizontal=True)
    view = view.sort_values(f"{sort.lower()}_surprise", ascending=False)
    st.dataframe(view[["people", "shared_films", "Critic surprise", "Audience surprise", "example_titles"]]
                 .rename(columns={"people": "People", "shared_films": "Shared films", "example_titles": "Example films"}),
                 hide_index=True, width="stretch")
    st.caption(f"{len(collabs)} pairs or groups. Critic range above 0: {(collabs['critic_low'] > 0).sum()}, below 0: "
               f"{(collabs['critic_high'] < 0).sum()}. Audience above 0: {(collabs['audience_low'] > 0).sum()}, below 0: "
               f"{(collabs['audience_high'] < 0).sum()}.")
