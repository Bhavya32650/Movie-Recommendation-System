
"""
CineMatch - Movie Recommendation System
Complete Streamlit Frontend with Light Theme

Run:
python -m streamlit run app/streamlit_app.py
"""

import os
import sys
import tempfile
import html

os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
from PIL import Image

from src.recommender import MovieRecommender
from src.model_training import train_and_save_pipeline


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# HTML RENDERING HELPER
# ==========================================================

def render_html(content):
    st.html(content)


# ==========================================================
# LIGHT THEME CSS
# ==========================================================

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --background: #F7F8FC;
    --panel: #FFFFFF;
    --border: #E7E8F0;
    --muted: #737B91;
    --purple: #7462CE;
    --teal: #159D91;
}

html, body {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F7F8FC;
    color: #27314F;
}

[data-testid="stHeader"] {
    background: rgba(255,255,255,.95);
}

[data-testid="stAppViewContainer"] {
    background: #F7F8FC;
}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E7E8F0;
}

[data-testid="stSidebar"] * {
    color: #303957;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #293453 !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #293453 !important;
}

p, label {
    color: #626B82;
}

/* HERO SECTION */

.hero {
    width: 100%;
    box-sizing: border-box;
    padding: 42px;
    border: 1px solid #E2DDF8;
    border-radius: 24px;
    background: linear-gradient(
        120deg,
        #EEE9FF 0%,
        #F6F4FF 48%,
        #E7F7F4 100%
    );
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "✦";
    position: absolute;
    right: 45px;
    top: -55px;
    font-size: 190px;
    color: rgba(116,98,206,.09);
    pointer-events: none;
}

.eyebrow {
    color: #7462CE;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(36px, 4vw, 56px);
    line-height: 1.12;
    font-weight: 700;
    color: #293453;
    margin: 15px 0;
}

.hero-copy {
    color: #626B82;
    font-size: 16px;
    line-height: 1.8;
    max-width: 750px;
}

.pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 25px;
}

.pill {
    display: inline-block;
    border: 1px solid #DDD7F5;
    background: rgba(255,255,255,.8);
    color: #6254B2;
    padding: 9px 15px;
    border-radius: 30px;
    font-size: 13px;
    font-weight: 600;
}

/* METRIC CARDS */

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E8E9F1;
    border-radius: 17px;
    padding: 20px;
    min-height: 110px;
    box-sizing: border-box;
    box-shadow: 0 4px 15px rgba(38,48,82,.04);
}

.metric-label {
    color: #7A8298;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    color: #303957;
    font-size: 32px;
    font-weight: 700;
    margin-top: 7px;
}

.metric-note {
    color: #159D91;
    font-size: 12px;
    margin-top: 5px;
}

/* SELECTED MOVIE CARD */

.movie-card {
    background: linear-gradient(145deg, #FFFFFF, #F5F2FF);
    border: 1px solid #E4DDF9;
    border-radius: 18px;
    padding: 25px;
    min-height: 230px;
    box-sizing: border-box;
    box-shadow: 0 5px 18px rgba(38,48,82,.05);
}

.movie-label {
    color: #7462CE;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.movie-title {
    color: #293453;
    font-size: 23px;
    font-weight: 700;
    line-height: 1.4;
    margin-bottom: 20px;
}

.movie-detail-label {
    color: #81899E;
    font-size: 12px;
    margin-top: 13px;
}

.movie-detail {
    color: #6153B0;
    font-size: 14px;
    margin-top: 4px;
    overflow-wrap: anywhere;
}

/* SECTION LABELS */

.kicker {
    color: #7462CE;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-description {
    color: #7A8298;
    font-size: 14px;
    margin-bottom: 20px;
}

/* TABS */

div[data-testid="stTabs"] button {
    color: #68718A !important;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7462CE !important;
    background: #F0EDFF;
    border-bottom: 3px solid #7462CE !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #E7E8F0;
}

/* INPUTS */

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] > div > div {
    background: #FFFFFF !important;
    border-color: #DCDDEA !important;
    border-radius: 11px !important;
}

div[data-testid="stSelectbox"] *,
div[data-testid="stNumberInput"] * {
    color: #303957;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(100deg, #7967D2, #6654BD);
    color: #FFFFFF !important;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    padding: 10px 15px;
    transition: .2s ease;
}

.stButton > button:hover {
    background: linear-gradient(100deg, #6654BD, #5140A6);
    color: #FFFFFF !important;
    transform: translateY(-1px);
}

/* TABLES */

[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #E7E8F0;
    border-radius: 13px;
    overflow: hidden;
}

/* STREAMLIT METRICS */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E7E8F0;
    padding: 15px;
    border-radius: 13px;
}

[data-testid="stMetricLabel"] {
    color: #737B91 !important;
}

[data-testid="stMetricValue"] {
    color: #293453 !important;
}

/* EXPANDERS */

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E7E8F0;
    border-radius: 12px;
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #858DA2;
    font-size: 12px;
    padding: 20px;
}

hr {
    border-color: #E7E8F0;
}

</style>
"""

render_html(CSS)


# ==========================================================
# LOAD RECOMMENDATION MODEL
# ==========================================================

@st.cache_resource
def load_recommender_model():

    model_path = os.path.join(
        ROOT,
        "models",
        "recommender_model.joblib"
    )

    if not os.path.exists(model_path):

        os.makedirs(
            os.path.dirname(model_path),
            exist_ok=True
        )

        with st.spinner(
            "Training recommendation model..."
        ):

            train_and_save_pipeline(
                data_dir=os.path.join(ROOT, "dataset"),
                model_output_path=model_path,
                min_movie_ratings=10,
                min_user_ratings=5,
                test_size=0.20,
                random_state=42
            )

    return MovieRecommender.load(model_path)


# ==========================================================
# METRIC CARD FUNCTION
# ==========================================================

def metric_card(label, value, note):

    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_note = html.escape(str(note))

    render_html(f"""
    <div class="metric-card">
        <div class="metric-label">{safe_label}</div>
        <div class="metric-value">{safe_value}</div>
        <div class="metric-note">{safe_note}</div>
    </div>
    """)


# ==========================================================
# MAIN APPLICATION
# ==========================================================

def main():

    # ------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------

    st.sidebar.markdown("## 🎛️ Discover Settings")

    st.sidebar.caption(
        "Customize your movie discovery experience."
    )

    top_k = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=25,
        value=10,
        step=1
    )

    method = st.sidebar.radio(
        "Recommendation Strategy",
        options=["item_based", "user_based"],
        format_func=lambda x:
            "Item-Based Filtering"
            if x == "item_based"
            else "User-Based Filtering",
        help="Choose the recommendation algorithm."
    )

    # ------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------

    try:
        model = load_recommender_model()

    except Exception as e:
        st.error(f"Unable to load model: {e}")
        st.info(
            "Check that your dataset and trained model files are available."
        )
        return

    # ------------------------------------------------------
    # SIDEBAR STATISTICS
    # ------------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Catalog Statistics")

    st.sidebar.metric(
        "Active Users",
        f"{len(model.user_to_idx):,}"
    )

    st.sidebar.metric(
        "Trained Movies",
        f"{len(model.movie_to_idx):,}"
    )

    st.sidebar.caption(
        "Powered by Collaborative Filtering"
    )

    # ------------------------------------------------------
    # HERO SECTION
    # ------------------------------------------------------

    render_html("""
    <div class="hero">

        <div class="eyebrow">
            YOUR PERSONAL CINEMA GUIDE
        </div>

        <div class="hero-title">
            Find your next<br>
            five-star movie.
        </div>

        <div class="hero-copy">
            Explore films with similar audiences, discover
            recommendations tailored to a user, and browse
            popular picks powered by collaborative filtering.
        </div>

        <div class="pills">
            <span class="pill">✦ Movie Similarity</span>
            <span class="pill">♟ Personalized Picks</span>
            <span class="pill">★ Popular Fallback</span>
            <span class="pill">▥ Model Insights</span>
        </div>

    </div>
    """)

    # ------------------------------------------------------
    # OVERVIEW CARDS
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "ACTIVE USERS",
            f"{len(model.user_to_idx):,}",
            "In trained dataset"
        )

    with col2:
        metric_card(
            "MOVIES",
            f"{len(model.movie_to_idx):,}",
            "Available to recommend"
        )

    with col3:
        metric_card(
            "TOP-K",
            str(top_k),
            "Selected result count"
        )

    st.write("")

    # ------------------------------------------------------
    # FOUR TABS
    # ------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔎 Explore Similar",
        "✨ Personalized Picks",
        "⭐ Popular Movies",
        "📊 Analytics"
    ])

    # ======================================================
    # TAB 1: MOVIE SIMILARITY
    # ======================================================

    with tab1:

        render_html("""
        <div class="kicker">01 / EXPLORE</div>
        """)

        st.header("Movies That Go Well Together")

        render_html("""
        <div class="section-description">
            Choose a movie and discover titles with the strongest
            co-rating patterns.
        </div>
        """)

        all_titles = sorted(
            model.movies_df["title"].dropna().unique().tolist()
        )

        if not all_titles:

            st.warning("No movies found in the catalog.")

        else:

            default_index = (
                all_titles.index("Toy Story (1995)")
                if "Toy Story (1995)" in all_titles
                else 0
            )

            selected_title = st.selectbox(
                "Search or Select a Movie",
                options=all_titles,
                index=default_index
            )

            movie_record = model.movies_df[
                model.movies_df["title"] == selected_title
            ].iloc[0]

            movie_id = int(movie_record["movieId"])

            col_a, col_b = st.columns(
                [1, 2],
                gap="large"
            )

            with col_a:

                st.markdown("#### 🎬 Selected Movie")

                safe_title = html.escape(str(selected_title))
                safe_genres = html.escape(str(movie_record["genres"]))

                render_html(f"""
                <div class="movie-card">

                    <div class="movie-label">
                        NOW EXPLORING
                    </div>

                    <div class="movie-title">
                        {safe_title}
                    </div>

                    <div class="movie-detail-label">
                        GENRES
                    </div>

                    <div class="movie-detail">
                        {safe_genres}
                    </div>

                    <div class="movie-detail-label">
                        MOVIE ID
                    </div>

                    <div class="movie-detail">
                        {movie_id}
                    </div>

                </div>
                """)

            with col_b:

                st.markdown("#### ✦ Similar Movies")

                with st.spinner("Finding similar movies..."):

                    similar_movies = model.recommend_similar_movies(
                        movie_id=movie_id,
                        top_k=top_k
                    )

                if similar_movies:

                    sim_df = pd.DataFrame(similar_movies)

                    sim_df.columns = [
                        "Movie ID",
                        "Title",
                        "Genres",
                        "Similarity Score"
                    ]

                    st.dataframe(
                        sim_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info("No similar movies found.")

    # ======================================================
    # TAB 2: PERSONALIZED RECOMMENDATIONS
    # ======================================================

    with tab2:

        render_html("""
        <div class="kicker">02 / PERSONALIZE</div>
        """)

        st.header("A Watchlist That Fits the User")

        render_html("""
        <div class="section-description">
            Enter a user ID to generate personalized movie recommendations.
        </div>
        """)

        col_input, col_button = st.columns(
            [2, 1],
            vertical_alignment="bottom"
        )

        with col_input:

            user_id = st.number_input(
                "Enter User ID",
                min_value=1,
                max_value=99999,
                value=1,
                step=1
            )

        with col_button:

            generate = st.button(
                "Generate My Picks ✨",
                use_container_width=True
            )

        user_id = int(user_id)

        if user_id in model.user_to_idx:

            watched = model.user_watched.get(
                user_id,
                set()
            )

            st.success(
                f"User {user_id} found. "
                f"{len(watched)} movies rated."
            )

            with st.expander("View User's Rating History"):

                u_idx = model.user_to_idx[user_id]
                user_row = model.sparse_matrix[u_idx]

                history = []

                for movie_idx, score in zip(
                    user_row.indices,
                    user_row.data
                ):

                    movie_id = model.idx_to_movie[movie_idx]

                    movie_data = model.movies_df[
                        model.movies_df["movieId"] == movie_id
                    ]

                    if not movie_data.empty:

                        history.append({
                            "Movie": movie_data.iloc[0]["title"],
                            "Rating": float(score),
                            "Genres": movie_data.iloc[0]["genres"]
                        })

                if history:

                    history_df = pd.DataFrame(history)

                    history_df = history_df.sort_values(
                        "Rating",
                        ascending=False
                    )

                    st.dataframe(
                        history_df.head(10),
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.caption("No rating history available.")

        else:

            st.info(
                f"User {user_id} is new. "
                "Popular movie fallback will be used."
            )

        if generate:

            with st.spinner("Generating recommendations..."):

                recommendations = model.recommend(
                    user_id=user_id,
                    method=method,
                    top_k=top_k
                )

            if recommendations:

                rec_df = pd.DataFrame(recommendations)

                display_cols = [
                    "title",
                    "genres",
                    "predicted_score",
                    "method"
                ]

                rename_map = {
                    "title": "Movie",
                    "genres": "Genres",
                    "predicted_score": "Predicted Rating",
                    "method": "Recommendation Method"
                }

                rec_df = rec_df[
                    [
                        col for col in display_cols
                        if col in rec_df.columns
                    ]
                ].rename(columns=rename_map)

                st.markdown(
                    f"#### Top {top_k} Recommendations"
                )

                st.dataframe(
                    rec_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    "Already-watched movies are excluded."
                )

            else:

                st.warning("No recommendations found.")

    # ======================================================
    # TAB 3: POPULAR MOVIES / COLD START
    # ======================================================

    with tab3:

        render_html("""
        <div class="kicker">03 / TRENDING</div>
        """)

        st.header("Popular Movies Worth a Look")

        render_html("""
        <div class="section-description">
            Discover popular movies recommended by the community.
        </div>
        """)

        render_html("""
        <div class="metric-card">

            <div style="
                font-size:19px;
                font-weight:700;
                color:#293453;
            ">
                ⭐ Why Popular Fallback?
            </div>

            <p style="
                color:#626B82;
                margin-top:12px;
                line-height:1.7;
            ">
                Collaborative filtering needs rating history.
                For a new user, the system uses popular movies
                ranked with a Bayesian weighted rating.
            </p>

        </div>
        """)

        popular_recs = model.recommend_popular(
            top_k=top_k
        )

        if popular_recs:

            popular_df = pd.DataFrame(popular_recs)

            columns = [
                "title",
                "genres",
                "predicted_score",
                "rating_count"
            ]

            popular_df = popular_df[
                [
                    col for col in columns
                    if col in popular_df.columns
                ]
            ]

            popular_df = popular_df.rename(
                columns={
                    "title": "Movie",
                    "genres": "Genres",
                    "predicted_score": "Bayesian Score",
                    "rating_count": "Rating Count"
                }
            )

            st.markdown("#### 🏆 Community Favorites")

            st.dataframe(
                popular_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("Popular movie data is not available.")

    # ======================================================
    # TAB 4: ANALYTICS
    # ======================================================

    with tab4:

        render_html("""
        <div class="kicker">04 / PERFORMANCE</div>
        """)

        st.header("Model Analytics and Evaluation")

        render_html("""
        <div class="section-description">
            Explore model evaluation metrics and data visualizations.
        </div>
        """)

        metric_cols = st.columns(4)

        metrics = [
            ("ITEM RMSE", "0.9193", "Lower is better"),
            ("ITEM MAE", "0.7021", "Lower is better"),
            ("USER PRECISION@10", "1.87%", "Ranking metric"),
            ("USER COVERAGE", "20.1%", "Catalog coverage")
        ]

        for col, (label, value, note) in zip(
            metric_cols,
            metrics
        ):

            with col:

                metric_card(
                    label,
                    value,
                    note
                )

        st.markdown("#### 📈 Exploratory Data Analysis")

        chart_files = [
            (
                "01_rating_distribution.png",
                "Rating Distribution"
            ),
            (
                "02_most_rated_movies.png",
                "Most-Rated Movies"
            ),
            (
                "03_highest_rated_movies.png",
                "Highest-Rated Movies"
            ),
            (
                "04_user_activity.png",
                "User Rating Activity"
            ),
            (
                "05_movie_popularity.png",
                "Movie Popularity"
            ),
            (
                "06_similarity_scores.png",
                "Cosine Similarity Heatmap"
            ),
            (
                "07_model_evaluation_comparison.png",
                "Model Evaluation Comparison"
            )
        ]

        chart_choice = st.selectbox(
            "Select a Chart",
            options=[
                name for _, name in chart_files
            ]
        )

        selected_file = next(
            filename
            for filename, name in chart_files
            if name == chart_choice
        )

        image_path = os.path.join(
            ROOT,
            "visualizations",
            selected_file
        )

        if os.path.exists(image_path):

            chart_image = Image.open(image_path)

            st.image(
                chart_image,
                caption=chart_choice,
                use_container_width=True
            )

        else:

            st.warning(
                f"Chart '{selected_file}' was not found. "
                "Run your chart-generation step."
            )

    # ======================================================
    # FOOTER
    # ======================================================

    st.markdown("---")

    render_html("""
    <div class="footer">
        🎬 CineMatch | Movie Discovery Powered by AI
        <br><br>
        Collaborative Filtering Recommendation System
    </div>
    """)


# ==========================================================
# APPLICATION ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
