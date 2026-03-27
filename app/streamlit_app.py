from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.heatmap import smooth_scalar_field


DATA_DIR = Path("data")
POEMS_PATH = DATA_DIR / "poems.parquet"
EMB_PATH = DATA_DIR / "embeddings.npy"
PROJ_PATH = DATA_DIR / "proj2d.npy"
POETS_PATH = DATA_DIR / "poet_centroids.parquet"
POET_COORDS_PATH = DATA_DIR / "poet_centroids_2d.npy"


@st.cache_data
def load_poem_artifacts() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    poems = pd.read_parquet(POEMS_PATH)
    emb = np.load(EMB_PATH)
    proj = np.load(PROJ_PATH)
    if len(poems) != emb.shape[0] or len(poems) != proj.shape[0]:
        raise ValueError("Artifact row count mismatch between poems, embeddings, and 2D projection")
    return poems, emb, proj


@st.cache_data
def load_poet_artifacts() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    if not POETS_PATH.exists() or not POET_COORDS_PATH.exists():
        return None, None
    poets = pd.read_parquet(POETS_PATH)
    coords = np.load(POET_COORDS_PATH)
    if len(poets) != coords.shape[0]:
        raise ValueError("Poet centroid metadata and coordinates do not match")
    return poets, coords


def init_state(n_poems: int) -> None:
    st.session_state.setdefault("rated_indices", [])
    st.session_state.setdefault("ratings", [])
    st.session_state.setdefault("current_index", 0 if n_poems > 0 else None)
    st.session_state.setdefault("last_result", None)


def reset_session(n_poems: int) -> None:
    st.session_state["rated_indices"] = []
    st.session_state["ratings"] = []
    st.session_state["current_index"] = 0 if n_poems > 0 else None
    st.session_state["last_result"] = None


def rate_current(value: float) -> None:
    idx = st.session_state.current_index
    if idx is None:
        return
    if idx not in st.session_state.rated_indices:
        st.session_state.rated_indices.append(idx)
        st.session_state.ratings.append(float(value))


def choose_next(embeddings: np.ndarray, mode: str, block_size: int) -> None:
    if not st.session_state.rated_indices:
        return
    rated_indices = np.array(st.session_state.rated_indices, dtype=np.int64)
    ratings = np.array(st.session_state.ratings, dtype=np.float64)
    result = run_blocked_step(embeddings, rated_indices, ratings, block_size=block_size)
    st.session_state.last_result = result
    st.session_state.current_index = result.exploit_index if mode == "exploit" else result.explore_index


def jump_to_title(poems: pd.DataFrame, query: str) -> None:
    if not query.strip():
        return
    mask = poems["title"].astype(str).str.contains(query, case=False, regex=False)
    matches = poems.index[mask].tolist()
    if matches:
        st.session_state.current_index = int(matches[0])


def plot_poem_space(poems: pd.DataFrame, proj2d: np.ndarray, overlay_mode: str) -> None:
    fig_data = None
    if st.session_state.last_result is not None:
        values = st.session_state.last_result.mean if overlay_mode == "preference" else st.session_state.last_result.variance
        fig_data = smooth_scalar_field(proj2d, values)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if fig_data is not None:
        ax.imshow(
            fig_data["zz"],
            extent=[fig_data["xs"][0], fig_data["xs"][-1], fig_data["ys"][0], fig_data["ys"][-1]],
            origin="lower",
            aspect="auto",
            alpha=0.75,
        )
    ax.scatter(proj2d[:, 0], proj2d[:, 1], s=4, alpha=0.16)
    if st.session_state.rated_indices:
        idx = np.array(st.session_state.rated_indices, dtype=int)
        ratings = np.array(st.session_state.ratings, dtype=float)
        liked = idx[ratings > 0]
        disliked = idx[ratings < 0]
        neutral = idx[ratings == 0]
        if len(liked):
            ax.scatter(proj2d[liked, 0], proj2d[liked, 1], s=20, alpha=0.9, label="liked")
        if len(disliked):
            ax.scatter(proj2d[disliked, 0], proj2d[disliked, 1], s=20, alpha=0.9, label="disliked")
        if len(neutral):
            ax.scatter(proj2d[neutral, 0], proj2d[neutral, 1], s=20, alpha=0.9, label="neutral")
    cur = st.session_state.current_index
    if cur is not None:
        ax.scatter([proj2d[cur, 0]], [proj2d[cur, 1]], s=90, marker="x", label="current")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Poem space")
    if st.session_state.rated_indices:
        ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig, clear_figure=True)


def plot_poet_space(poets: pd.DataFrame | None, poet_coords: np.ndarray | None, topn: int) -> None:
    if poets is None or poet_coords is None or len(poets) == 0:
        st.info("Poet centroid artifacts not found yet. Build them with scripts/build_poet_centroids.py and scripts/project_poet_centroids_2d.py.")
        return
    poets = poets.sort_values("n_poems", ascending=False).head(topn).reset_index(drop=True)
    coords = poet_coords[: len(poets)]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sizes = 20 + 8 * np.sqrt(poets["n_poems"].to_numpy())
    ax.scatter(coords[:, 0], coords[:, 1], s=sizes, alpha=0.5)
    label_n = min(20, len(poets))
    for i in range(label_n):
        ax.text(coords[i, 0], coords[i, 1], str(poets.iloc[i]["poet"]), fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Poet centroid map")
    st.pyplot(fig, clear_figure=True)


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Poetry GP Explorer")
    st.caption("Interactive exploration of poem embedding space with an exact GP preference model.")

    poems, embeddings, proj2d = load_poem_artifacts()
    poets, poet_coords = load_poet_artifacts()
    init_state(len(poems))

    with st.sidebar:
        st.header("Controls")
        view_mode = st.radio("View", ["poem space", "poet space"], index=0)
        overlay_mode = st.radio("Overlay", ["preference", "uncertainty"], horizontal=True)
        block_size = st.selectbox("Blocked backend block size", [256, 512, 1024, 2048, 4096], index=3)
        poet_topn = st.slider("Poets shown on map", min_value=20, max_value=300, value=80, step=10)
        if st.button("Reset session"):
            reset_session(len(poems))
        search_query = st.text_input("Jump to poem by title substring")
        if st.button("Jump to title"):
            jump_to_title(poems, search_query)
        st.markdown("---")
        st.markdown("Run heavy jobs from a compute allocation, not the login node.")

    left, right = st.columns([1.4, 1.0])

    with left:
        if view_mode == "poem space":
            plot_poem_space(poems, proj2d, overlay_mode)
        else:
            plot_poet_space(poets, poet_coords, poet_topn)

    with right:
        st.subheader("Current poem")
        cur = st.session_state.current_index
        if cur is None:
            st.write("No current poem selected.")
        else:
            row = poems.iloc[cur]
            st.markdown(f"**{row['title']}**")
            st.markdown(f"*{row['poet']}*")
            st.caption(f"poem_id: {row['poem_id']}")
            st.text_area("Text", str(row["text"]), height=260)

            c1, c2, c3 = st.columns(3)
            if c1.button("Dislike"):
                rate_current(-1.0)
            if c2.button("Neutral"):
                rate_current(0.0)
            if c3.button("Like"):
                rate_current(1.0)

            c4, c5 = st.columns(2)
            if c4.button("Exploit next"):
                choose_next(embeddings, "exploit", block_size)
            if c5.button("Explore next"):
                choose_next(embeddings, "explore", block_size)

        st.subheader("Rated poems")
        if st.session_state.rated_indices:
            rated_df = poems.iloc[st.session_state.rated_indices].copy()
            rated_df["rating"] = st.session_state.ratings
            show_cols = [c for c in ["poem_id", "title", "poet", "rating"] if c in rated_df.columns or c == "rating"]
            st.dataframe(rated_df[show_cols], use_container_width=True)
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
