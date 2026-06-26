"""Steam Discovery Engine — Streamlit dashboard.

Thin UI client that talks to the FastAPI service over HTTP:
    Streamlit UI  ->  HTTP (REST)  ->  FastAPI  ->  Recommendation Pipeline

Run:
    uvicorn src.api.app:app --port 8000      # start the API first
    streamlit run dashboard/app.py            # then the dashboard

Тонкий UI-клиент, который ходит в FastAPI по HTTP.
"""
import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
BEST_RECALL = 0.1484  # project's best result (Two-Stage Recommendation, Recall@10)

st.set_page_config(page_title="Steam Discovery Engine", page_icon="🎮", layout="wide")

# Bigger, more readable sidebar menu
# Более крупное и читаемое меню слева
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 16px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path, params=None):
    # Call the API; return (ok, data, elapsed_ms) without crashing the UI
    # Запрос к API; возвращаем (ok, data, elapsed_ms) без падения UI
    try:
        t0 = time.perf_counter()
        r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            return True, r.json(), ms
        return False, r.json().get("detail", f"HTTP {r.status_code}"), ms
    except requests.RequestException as e:
        return False, f"API unavailable at {API_URL} ({e})", 0.0


def footer():
    st.divider()
    st.caption("Steam Discovery Engine · Two-Stage Recommendation System · "
               "Built with FastAPI · LightGBM · FAISS · Streamlit · Docker")


def rec_card(rec):
    with st.container(border=True):
        st.markdown(f"**{rec['title']}**")
        if rec.get("genres"):
            st.caption(rec["genres"].replace(",", " • "))
        st.markdown(f"Match score&nbsp;&nbsp;`{rec['score']:.3f}`", unsafe_allow_html=True)
        st.markdown(f"_{rec['reason']}_")


def similar_card(g):
    with st.container(border=True):
        st.markdown(f"**{g['title']}**")
        st.markdown(f"Similarity&nbsp;&nbsp;`{g['score']:.3f}`", unsafe_allow_html=True)


# --- Sidebar navigation ---
st.sidebar.title("🎮 Steam Discovery Engine")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🎯 Recommendations", "🔍 Similar Games", "⚙️ How it Works", "🏗 Architecture"],
)
st.sidebar.caption(f"API: {API_URL}")


def page_home():
    st.title("Steam Discovery Engine")
    st.subheader("Two-Stage Recommendation System")
    ok, info, _ = api_get("/")
    n_games = info.get("games", "—") if ok else "—"
    n_users = info.get("users", "—") if ok else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@10", f"{BEST_RECALL:.4f}")
    c2.metric("Models", 5)
    c3.metric("Games", f"{n_games:,}" if isinstance(n_games, int) else n_games)
    c4.metric("Users", f"{n_users:,}" if isinstance(n_users, int) else n_users)

    if not ok:
        st.warning(f"API not reachable at {API_URL}. Start it with "
                   "`uvicorn src.api.app:app --port 8000`.")

    st.markdown("#### Pipeline")
    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            node [shape=box, style=rounded, fontsize=10];
            Interactions -> "Retrieval Models" -> "Candidate Fusion"
              -> LambdaMART -> "Top-10" -> FastAPI -> Dashboard;
        }
        """
    )
    st.markdown(
        "Classical, semantic and neural retrieval models generate candidates, which a "
        "LambdaMART re-ranker orders into the final Top-10. Use the left menu to try it."
    )
    st.markdown(f"[Open API docs (Swagger)]({API_URL}/docs)")
    footer()


def page_recommendations():
    st.title("Recommendations")
    mode = st.radio("Choose a user", ["Demo profile", "Steam User ID"], horizontal=True)

    user_id = None
    if mode == "Demo profile":
        ok, users, _ = api_get("/demo/users")
        if not ok:
            st.error(users)
            return
        names = {u["display_name"]: u for u in users}
        choice = st.selectbox("Demo profile", list(names))
        u = names[choice]
        user_id = u["user_id"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Favorite genre", u["favorite_genre"])
        c2.metric("Games", u["games"])
        c3.metric("Avg playtime (h)", u["avg_playtime"])
    else:
        ok, users, _ = api_get("/demo/users")
        if "uid" not in st.session_state:
            st.session_state.uid = ""
        st.text_input("Steam User ID", key="uid", placeholder="e.g. 76561198840684331")
        if ok:
            with st.expander("Show demo users"):
                for u in users:
                    st.button(
                        f"{u['display_name']} — {u['user_id']}",
                        key=f"use_{u['user_id']}",
                        on_click=lambda v=u["user_id"]: st.session_state.update(uid=str(v)),
                    )
        raw = st.session_state.uid
        user_id = int(raw) if raw.strip().isdigit() else None

    if st.button("Generate Recommendations", type="primary") and user_id is not None:
        ok, data, ms = api_get(f"/recommend/{user_id}")
        if not ok:
            st.error(data)
            return
        st.markdown("### Top-10 Recommendations")
        st.caption(f"Ranked by LambdaMART · generated in {ms:.0f} ms")
        cols = st.columns(2)
        for i, rec in enumerate(data["recommendations"]):
            with cols[i % 2]:
                rec_card(rec)
    footer()


def page_similar():
    st.title("Similar Games")
    st.caption("Item-to-item retrieval in the semantic embedding space.")
    ok, games, _ = api_get("/demo/games")
    if not ok:
        st.error(games)
        return
    titles = {g["title"]: g["game_id"] for g in games}
    choice = st.selectbox("Pick a game", list(titles))
    gid = titles[choice]

    if st.button("Find Similar", type="primary"):
        ok, data, _ = api_get(f"/similar/{gid}")
        if not ok:
            st.error(data)
            return
        st.markdown(f"### Games similar to {data['game']}")
        cols = st.columns(2)
        for i, g in enumerate(data["similar"]):
            with cols[i % 2]:
                similar_card(g)
    footer()


def page_how():
    st.title("How it Works")
    st.markdown("Every recommendation goes through five steps:")
    st.markdown(
        "**1. Retrieve candidates** — ALS, Semantic, Two-Tower and Popularity each "
        "propose relevant games.\n\n"
        "**2. Merge candidates** — their suggestions are fused into one candidate set.\n\n"
        "**3. Build ranking features** — retrieval scores, genre/tag overlap, "
        "popularity, price and user stats.\n\n"
        "**4. LambdaMART re-ranks** — a learning-to-rank model orders the candidates.\n\n"
        "**5. Return Top-10** — the best games are returned with a reason."
    )
    st.info("Retrieval finds *relevant* candidates; re-ranking decides their *order*. "
            "No single model is trusted alone — the ranker blends them.")
    footer()


def page_architecture():
    st.title("Architecture")
    st.graphviz_chart(
        """
        digraph G {
            rankdir=TB;
            node [shape=box, style=rounded, fontsize=10];

            subgraph cluster_offline {
                label="Offline Training"; color="#4C72B0"; fontcolor="#4C72B0";
                Interactions;
                ALS; Semantic; TwoTower; Popularity;
                Fusion [label="Candidate Fusion"];
                Features [label="Feature Pipeline"];
                LambdaMART;
            }
            subgraph cluster_online {
                label="Online Inference"; color="#55A868"; fontcolor="#55A868";
                FastAPI; Streamlit;
            }

            Interactions -> ALS;
            Interactions -> Semantic;
            Interactions -> TwoTower;
            Interactions -> Popularity;
            ALS -> Fusion; Semantic -> Fusion; TwoTower -> Fusion; Popularity -> Fusion;
            Fusion -> Features -> LambdaMART;
            LambdaMART -> FastAPI [label="exported models"];
            FastAPI -> Streamlit [label="REST"];
        }
        """
    )
    st.caption("Cheap retrieval narrows the catalog; an expensive ranker orders the "
               "survivors; FastAPI serves them; Streamlit presents them.")
    footer()


PAGES = {
    "🏠 Home": page_home,
    "🎯 Recommendations": page_recommendations,
    "🔍 Similar Games": page_similar,
    "⚙️ How it Works": page_how,
    "🏗 Architecture": page_architecture,
}
PAGES[page]()
