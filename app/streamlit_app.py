from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.heatmap import smooth_scalar_field


DATA_DIR = Path("data")
POEMS_PATH = DATA_DIR / "poems.parquet"
EMB_PATH = DATA_DIR / "embeddings.npy"
PROJ_PATH = DATA_DIR / "proj2d.npy"
TEXT_CANDIDATES = ["text", "poem", "content", "body"]
TITLE_CANDIDATES = ["title", "poem_title"]
POET_CANDIDATES = ["poet", "author"]


@st.cache_data
def load_artifacts() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    poems = pd.read_parquet(POEMS_PATH)
    emb = np.load(EMB_PATH)
    proj = np.load(PROJ_PATH)
    if len(poems) != emb.shape[0] or len(poems) != proj.shape[0]:
        raise ValueError("Artifact row count mismatch between poems, embeddings, and 2D projection")
    return poems, emb, proj


def pick_column(columns: list[str], candidates: list[str], fallback: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    return fallback


def init_state(n_poems: int) -> None:
    st.session_state.setdefault("rated_indices", [])
    st.session_state.setdefault("ratings", [])
    st.session_state.setdefault("current_index", 0 if n_poems > 0 else None)
    st.session_state.setdefault("last_result", None)


def rate_current(value: float) -> None:
    idx = st.session_state.current_index
    if idx is None:
        return
    if idx not in st.session_state.rated_indices:
        st.session_state.rated_indices.append(idx)
        st.session_state.ratings.append(float(value))


def choose_next(embeddings: np.ndarray, mode: str) -> None:
    if not st.session_state.rated_indices:
        return
    rated_indices = np.array(st.session_state.rated_indices, dtype=np.int64)
    ratings = np.array(st.session_state.ratings, dtype=np.float64)
    result = run_blocked_step(embeddings, rated_indices, ratings)
    st.session_state.last_result = result
    st.session_state.current_index = result.exploit_index if mode == "exploit" else result.explore_index


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Poetry GP Explorer")

    poems, embeddings, proj2d = load_artifacts()
    init_state(len(poems))

    text_col = pick_column(list(poems.columns), TEXT_CANDIDATES, poems.columns[-1])
    title_col = pick_column(list(poems.columns), TITLE_CANDIDATES, poems.columns[0])
    poet_col = pick_column(list(poems.columns), POET_CANDIDATES, poems.columns[0])

    left, right = st.columns([1.35, 1.0])

    with left:
        st.subheader("Poem space")
        fig_data = None
        if st.session_state.last_result is not None:
            mode = st.radio("Overlay", ["preference", "uncertainty"], horizontal=True)
            values = st.session_state.last_result.mean if mode == "preference" else st.session_state.last_result.variance
            fig_data = smooth_scalar_field(proj2d, values)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        if fig_data is not None:
            ax.imshow(
                fig_data["zz"],
                extent=[fig_data["xs"][0], fig_data["xs"][-1], fig_data["ys"][0], fig_data["ys"][-1]],
                origin="lower",
                aspect="auto",
                alpha=0.75,
            )
        ax.scatter(proj2d[:, 0], proj2d[:, 1], s=4, alpha=0.18)
        if st.session_state.rated_indices:
            idx = np.array(st.session_state.rated_indices, dtype=int)
            ax.scatter(proj2d[idx, 0], proj2d[idx, 1], s=20, alpha=0.9)
        cur = st.session_state.current_index
        if cur is not None:
            ax.scatter([proj2d[cur, 0]], [proj2d[cur, 1]], s=80, marker="x")
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Current poem")
        cur = st.session_state.current_index
        if cur is None:
            st.write("No current poem selected.")
        else:
            row = poems.iloc[cur]
            st.markdown(f"**{row[title_col]}**")
            st.markdown(f"*{row[poet_col]}*")
            st.text_area("Text", str(row[text_col]), height=280)
            c1, c2, c3 = st.columns(3)
            if c1.button("Dislike"):
                rate_current(-1.0)
            if c2.button("Neutral"):
                rate_current(0.0)
            if c3.button("Like"):
                rate_current(1.0)
            c4, c5 = st.columns(2)
            if c4.button("Exploit next"):
                choose_next(embeddings, "exploit")
            if c5.button("Explore next"):
                choose_next(embeddings, "explore")

        st.subheader("Rated poems")
        if st.session_state.rated_indices:
            rated_df = poems.iloc[st.session_state.rated_indices].copy()
            rated_df["rating"] = st.session_state.ratings
            cols = [c for c in [title_col, poet_col, "rating"] if c in rated_df.columns or c == "rating"]
            st.dataframe(rated_df[cols], use_container_width=True)
        else:
            st.write("No ratings yet.")

        if st.session_state.last_result is not None:
            prof = st.session_state.last_result.profile
            st.subheader("Last step timing")
            st.json(
                {
                    "fit_seconds": prof.fit_seconds,
                    "score_seconds": prof.score_seconds,
                    "select_seconds": prof.select_seconds,
                    "total_seconds": prof.total_seconds,
                }
            )


if __name__ == "__main__":
    main()
