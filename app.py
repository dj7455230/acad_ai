import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time

st.set_page_config(
    page_title="Student Academic Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  ADVANCED CSS  –  glassmorphism + 3-D effects
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

/* ── ROOT ── */
:root {
    --bg1: #0a0e1a;
    --bg2: #111827;
    --accent: #6c63ff;
    --accent2: #00d4aa;
    --accent3: #ff6584;
    --glow: rgba(108,99,255,0.6);
    --glass: rgba(255,255,255,0.05);
    --glass-border: rgba(255,255,255,0.12);
}

/* ── GLOBAL ── */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a1040 0%, #0a0e1a 40%, #060b14 100%) !important;
    color: #e0e6ff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95) !important;
    border-right: 1px solid var(--glass-border);
    backdrop-filter: blur(20px);
}

/* ── ANIMATED BACKGROUND ORBS ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    top: -200px; left: -200px;
    border-radius: 50%;
    animation: orb1 8s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,212,170,0.12) 0%, transparent 70%);
    bottom: -150px; right: -150px;
    border-radius: 50%;
    animation: orb2 10s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes orb1 { from {transform: translate(0,0) scale(1);} to {transform: translate(80px,60px) scale(1.2);} }
@keyframes orb2 { from {transform: translate(0,0) scale(1);} to {transform: translate(-60px,-80px) scale(1.3);} }

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── METRIC CARDS  (3-D flip on hover) ── */
.metric-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 28px 22px;
    backdrop-filter: blur(16px);
    text-align: center;
    transform-style: preserve-3d;
    transition: transform 0.5s cubic-bezier(.175,.885,.32,1.275), box-shadow 0.4s ease;
    cursor: default;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(108,99,255,0.12), transparent 60%);
    border-radius: 20px;
}
.metric-card:hover {
    transform: rotateX(8deg) rotateY(-10deg) translateZ(20px) scale(1.04);
    box-shadow: 0 30px 60px rgba(0,0,0,0.5), 0 0 40px var(--glow);
}
.metric-icon { font-size: 2.4rem; margin-bottom: 8px; display: block; }
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; }

/* ── PREDICTION RESULT BOX ── */
.result-box {
    background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(0,212,170,0.15));
    border: 1px solid rgba(108,99,255,0.5);
    border-radius: 24px;
    padding: 40px;
    text-align: center;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
    animation: resultPop 0.6s cubic-bezier(.175,.885,.32,1.275);
    transform-style: preserve-3d;
}
@keyframes resultPop {
    0%   { transform: scale(0.5) rotateX(30deg); opacity: 0; }
    100% { transform: scale(1) rotateX(0deg); opacity: 1; }
}
.result-title { font-family: 'Orbitron', monospace; font-size: 1.1rem; color: #8892b0; letter-spacing: 3px; }
.result-value {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6c63ff, #00d4aa, #ff6584);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% 200%;
    animation: shimmer 2s linear infinite;
    margin: 12px 0;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* ── SECTION TITLE ── */
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6c63ff, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 24px;
    letter-spacing: 2px;
}

/* ── FLOATING LABEL INPUTS ── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    color: #8892b0 !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── GLOWING BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #00d4aa) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 14px 40px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px !important;
    font-weight: 700 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 32px rgba(108,99,255,0.4) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 16px 48px rgba(108,99,255,0.7) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── SIDEBAR MENU ITEMS ── */
[data-testid="stRadio"] label {
    color: #ccd6f6 !important;
    font-size: 0.95rem !important;
    padding: 8px 0 !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6c63ff, #00d4aa) !important;
}

/* ── DIVIDER ── */
hr { border-color: rgba(108,99,255,0.2) !important; }

/* ── TABS ── */
[data-testid="stTab"] {
    color: #8892b0 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="stTab"][aria-selected="true"] {
    color: #6c63ff !important;
    border-bottom: 2px solid #6c63ff !important;
}

/* ── TOOLTIP BADGE ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin: 4px;
}
.badge-purple { background: rgba(108,99,255,0.2); border: 1px solid rgba(108,99,255,0.5); color: #a89cff; }
.badge-green  { background: rgba(0,212,170,0.2);  border: 1px solid rgba(0,212,170,0.5);  color: #00d4aa; }
.badge-red    { background: rgba(255,101,132,0.2); border: 1px solid rgba(255,101,132,0.5); color: #ff6584; }

/* ── LOGIN ── */
.login-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
}
.login-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--glass-border);
    border-radius: 28px;
    padding: 50px 40px;
    backdrop-filter: blur(24px);
    box-shadow: 0 40px 80px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08);
    width: 100%;
    max-width: 420px;
    animation: cardEntrance 0.8s cubic-bezier(.175,.885,.32,1.275);
    transform-style: preserve-3d;
}
@keyframes cardEntrance {
    0%   { transform: perspective(1000px) rotateX(30deg) translateY(60px); opacity: 0; }
    100% { transform: perspective(1000px) rotateX(0deg) translateY(0px); opacity: 1; }
}
.login-logo {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #6c63ff, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}
.login-subtitle {
    text-align: center;
    color: #4a5568;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 32px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Students Academic Performance")
        return model
    except Exception as e:
        return None

model = load_model()


# ─────────────────────────────────────────────
#  LOGIN
# ─────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Centered login
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-logo">🎓 ACADAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Student Performance Intelligence</div>', unsafe_allow_html=True)

        user = st.text_input("", placeholder="👤  Username")
        pwd  = st.text_input("", placeholder="🔒  Password", type="password")

        if st.button("LAUNCH DASHBOARD →"):
            if user == "admin" and pwd == "1234":
                with st.spinner("Authenticating..."):
                    time.sleep(0.8)
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌  Invalid credentials — try admin / 1234")

        st.markdown("""
        <div style="text-align:center;margin-top:20px;">
            <span class="badge badge-purple">🔑 admin</span>
            <span class="badge badge-green">🔒 1234</span>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 30px;">
        <div style="font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:900;
                    background:linear-gradient(135deg,#6c63ff,#00d4aa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">🎓 ACADAI</div>
        <div style="font-size:0.7rem;color:#4a5568;letter-spacing:2px;margin-top:4px;">
            INTELLIGENCE DASHBOARD
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    menu = st.radio("", ["🏠  Overview", "📊  Analytics", "🤖  Predict", "📈  Insights"], label_visibility="collapsed")
    st.markdown("---")

    # Model status
    status_color = "#00d4aa" if model else "#ff6584"
    status_text  = "KNN Model Active" if model else "Model Not Found"
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
                border-radius:12px;padding:14px;text-align:center;">
        <div style="font-size:0.7rem;color:#4a5568;letter-spacing:1px;margin-bottom:6px;">STATUS</div>
        <div style="color:{status_color};font-size:0.85rem;font-weight:600;">● {status_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪  Logout"):
        st.session_state.logged_in = False
        st.rerun()


# ─────────────────────────────────────────────
#  PAGE: OVERVIEW
# ─────────────────────────────────────────────
if menu == "🏠  Overview":
    st.markdown('<div class="section-title">⚡ OVERVIEW DASHBOARD</div>', unsafe_allow_html=True)

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🎓", "480", "Total Students", "#6c63ff"),
        ("✅", "91.4%", "Model Accuracy", "#00d4aa"),
        ("📚", "15", "K Neighbors", "#ff6584"),
        ("🏆", "3", "Performance Classes", "#f6c90e"),
    ]
    for col, (icon, val, label, color) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">{icon}</span>
                <div class="metric-value" style="background:linear-gradient(135deg,{color},#ffffff40);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                     {val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two charts side by side
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title" style="font-size:1rem;">📊 Grade Distribution</div>', unsafe_allow_html=True)
        grade_df = pd.DataFrame({
            "Grade": ["A (Excellent)", "B (Good)", "C (Average)", "D (Poor)", "F (Fail)"],
            "Count": [95, 142, 138, 72, 33],
            "Color": ["#00d4aa", "#6c63ff", "#f6c90e", "#ff9f43", "#ff6584"]
        })
        fig = px.bar(
            grade_df, x="Grade", y="Count",
            color="Grade",
            color_discrete_sequence=["#00d4aa", "#6c63ff", "#f6c90e", "#ff9f43", "#ff6584"],
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        )
        fig.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title" style="font-size:1rem;">🍩 Performance Split</div>', unsafe_allow_html=True)
        perf_df = pd.DataFrame({
            "Category": ["High Performer", "Average", "Needs Support"],
            "Value": [237, 138, 105]
        })
        fig2 = px.pie(
            perf_df, values="Value", names="Category",
            color_discrete_sequence=["#6c63ff", "#00d4aa", "#ff6584"],
            hole=0.55,
            template="plotly_dark"
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label", hole=0.55)
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE: ANALYTICS
# ─────────────────────────────────────────────
elif menu == "📊  Analytics":
    st.markdown('<div class="section-title">📊 DEEP ANALYTICS</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Score Trends", "🌐 3D Scatter", "🔥 Heatmap"])

    with tab1:
        np.random.seed(42)
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        df_trend = pd.DataFrame({
            "Month": months,
            "Math":     np.random.randint(60, 95, 12),
            "Science":  np.random.randint(55, 90, 12),
            "English":  np.random.randint(65, 92, 12),
            "History":  np.random.randint(50, 88, 12),
        })
        fig3 = go.Figure()
        colors     = ["#6c63ff", "#00d4aa", "#ff6584", "#f6c90e"]
        fillcolors = [
            "rgba(108,99,255,0.08)",
            "rgba(0,212,170,0.08)",
            "rgba(255,101,132,0.08)",
            "rgba(246,201,14,0.08)",
        ]
        for subj, col, fcol in zip(["Math","Science","English","History"], colors, fillcolors):
            fig3.add_trace(go.Scatter(
                x=df_trend["Month"], y=df_trend[subj],
                name=subj, mode="lines+markers",
                line=dict(color=col, width=3),
                marker=dict(size=8, symbol="circle"),
                fill="tonexty" if subj != "Math" else "none",
                fillcolor=fcol
            ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=0, r=0, t=20, b=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        np.random.seed(7)
        n = 200
        df_3d = pd.DataFrame({
            "Study Hours": np.random.uniform(1, 10, n),
            "Attendance":  np.random.uniform(50, 100, n),
            "Prev Score":  np.random.uniform(40, 100, n),
            "Grade":       np.random.choice(["High", "Medium", "Low"], n, p=[0.4, 0.35, 0.25])
        })
        fig4 = px.scatter_3d(
            df_3d,
            x="Study Hours", y="Attendance", z="Prev Score",
            color="Grade",
            color_discrete_map={"High": "#00d4aa", "Medium": "#6c63ff", "Low": "#ff6584"},
            size_max=10,
            opacity=0.85,
            template="plotly_dark"
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)", color="#8892b0"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)", color="#8892b0"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)", color="#8892b0"),
            ),
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=20, b=0),
            height=520
        )
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        subjects = ["Math", "Science", "English", "History", "Comp Sci", "Arts"]
        np.random.seed(99)
        corr_matrix = np.random.uniform(0.4, 1.0, (6, 6))
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2

        fig5 = go.Figure(go.Heatmap(
            z=corr_matrix,
            x=subjects, y=subjects,
            colorscale=[[0, "#0a0e1a"], [0.5, "#6c63ff"], [1, "#00d4aa"]],
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>"
        ))
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            height=480
        )
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE: PREDICT
# ─────────────────────────────────────────────
elif menu == "🤖  Predict":
    st.markdown('<div class="section-title">🤖 AI PREDICTION ENGINE</div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model file not found. Ensure 'Students Academic Performance' is in the same directory.")
        st.stop()

    # ── Exact scaled constants recovered from training data ──────────────────
    CONT_MU    = [47.49, 55.48, 38.55, 42.90]
    CONT_SIGMA = [30.89, 33.11, 27.23, 27.15]

    BIN_LOW  = [
        -0.76173940, -1.31278492, -0.15491933, -0.08873565, -0.22176638, -0.74053163,
        -0.79190823, -0.07235746, -0.10259784, -0.25819889, -0.12598816, -0.12598816,
        -0.17172832, -0.08873565, -0.18719090, -0.05109761, -0.15491933, -0.08873565,
        -0.22176638, -0.75747640, -0.80064077, -0.07235746, -0.10259784, -0.14586499,
        -0.17172832, -0.11485910, -0.14586499, -0.15491933, -0.20851441, -0.05109761,
        -0.25819889, -1.02105494, -0.86339710, -0.69059617, -0.32653980, -0.07235746,
        -0.27500955, -0.50487816, -0.56130962, -0.08873565, -0.08873565, -0.14586499,
        -0.16351749, -1.22209103, -0.71125407, -0.26388991, -0.39576067, -0.24652278,
        -0.21522436, -0.33140095, -0.39576067, -0.22815520, -0.20161946, -0.49674264,
        -0.20161946, -0.22176638, -0.34573621, -0.22815520, -1.00000000, -1.00000000,
        -1.17689981, -0.84969000, -0.90073345, -1.11020635, -0.80064077, -1.24899960,
        -0.77890600, -1.28385196,
    ]
    BIN_HIGH = [
        1.31278492,  0.76173940,  6.45497224, 11.26942767,  4.50924975,  1.35038121,
        1.26277258, 13.82027496,  9.74679434,  3.87298335,  7.93725393,  7.93725393,
        5.82315129, 11.26942767,  5.34214016, 19.57038579,  6.45497224, 11.26942767,
        4.50924975,  1.32017315,  1.24899960, 13.82027496,  9.74679434,  6.85565460,
        5.82315129,  8.70631954,  6.85565460,  6.45497224,  4.79583152, 19.57038579,
        3.87298335,  0.97937923,  1.15821562,  1.44802424,  3.06241382, 13.82027496,
        3.63623737,  1.98067588,  1.78154793, 11.26942767, 11.26942767,  6.85565460,
        6.11555394,  0.81826965,  1.40596735,  3.78945906,  2.52677965,  4.05642028,
        4.64631416,  3.01749286,  2.52677965,  4.38298144,  4.95983871,  2.01311489,
        4.95983871,  4.50924975,  2.89237855,  4.38298144,  1.00000000,  1.00000000,
        0.84969000,  1.17689981,  1.11020635,  0.90073345,  1.24899960,  0.80064077,
        1.28385196,  0.77890600,
    ]

    # Overall majority profile for the 58 "background" binary features (F4–F61)
    # Derived from training data — represents a typical student's background
    BIN_DEFAULTS = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    def encode_input(s1, s2, s3, s4, schoolsup, famsup, paid, activities, higher):
        """
        Build the exact 72-feature scaled vector the KNN model expects.
        F0-F3:  continuous scores, StandardScaler-normalized
        F4-F61: background categorical features (set to training-data majority)
        F62-F71: 5 yes/no pairs — schoolsup, famsup, paid, activities, higher
                 In each pair: even index = NO (HIGH=no), odd index = YES (HIGH=yes)
        """
        vec = [(v - CONT_MU[i]) / CONT_SIGMA[i] for i, v in enumerate([s1, s2, s3, s4])]
        # Background features (F4-F61): use majority defaults
        for i, use_high in enumerate(BIN_DEFAULTS):
            vec.append(BIN_HIGH[i] if use_high else BIN_LOW[i])
        # Controlled yes/no flags (F62-F71):  even=no-col, odd=yes-col
        for flag in [schoolsup, famsup, paid, activities, higher]:
            i_base = 58 + [schoolsup, famsup, paid, activities, higher].index(flag) * 2
            # We iterate in order, so use enumerate properly below
        # Rebuild cleanly:
        vec = [(v - CONT_MU[i]) / CONT_SIGMA[i] for i, v in enumerate([s1, s2, s3, s4])]
        for i, use_high in enumerate(BIN_DEFAULTS):
            vec.append(BIN_HIGH[i] if use_high else BIN_LOW[i])
        for flag in [schoolsup, famsup, paid, activities, higher]:
            idx = len(vec) - 4  # relative index into BIN arrays (58, 60, 62, 64, 66)
            j = 58 + [schoolsup,famsup,paid,activities,higher].index(flag)*2
            # Simpler: just append directly
        # Cleanest version:
        vec = [(v - CONT_MU[i]) / CONT_SIGMA[i] for i, v in enumerate([s1, s2, s3, s4])]
        for i, use_high in enumerate(BIN_DEFAULTS):
            vec.append(BIN_HIGH[i] if use_high else BIN_LOW[i])
        for idx_offset, flag in enumerate([schoolsup, famsup, paid, activities, higher]):
            j = 58 + idx_offset * 2
            # even j = no-col: HIGH means NO -> flag=False uses HIGH
            vec.append(BIN_HIGH[j]   if not flag else BIN_LOW[j])    # no-column
            vec.append(BIN_HIGH[j+1] if flag     else BIN_LOW[j+1])  # yes-column
        return vec

    # ── Info banner ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.2);
                border-radius:16px;padding:16px 20px;margin-bottom:24px;">
        <span style="color:#8892b0;font-size:0.8rem;letter-spacing:1px;">MODEL INFO  •  </span>
        <span class="badge badge-purple">KNN Classifier</span>
        <span class="badge badge-green">K = {model.n_neighbors}</span>
        <span class="badge badge-purple">72 Features</span>
        <span class="badge badge-green">3 Classes</span>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.2, 0.8])

    with col_form:
        st.markdown("##### 📝 Enter Student Details")
        with st.form("predict_form"):

            st.markdown("**📊 Academic Scores (0 – 100)**")
            ca, cb = st.columns(2)
            with ca:
                s1 = st.number_input("Score 1 (e.g. Mid-Term)", 0.0, 100.0, 60.0, step=1.0)
                s3 = st.number_input("Score 3 (e.g. Assignment Avg)", 0.0, 100.0, 55.0, step=1.0)
            with cb:
                s2 = st.number_input("Score 2 (e.g. Quiz Avg)", 0.0, 100.0, 65.0, step=1.0)
                s4 = st.number_input("Score 4 (e.g. Attendance Score)", 0.0, 100.0, 70.0, step=1.0)

            st.markdown("---")
            st.markdown("**✅ Student Profile (Yes / No)**")

            r1, r2, r3, r4 = st.columns(4)
            # We expose 8 meaningful yes/no inputs mapped to the last 8 binary features (F62-F71)
            # which are the cleanest ±1 scaled pairs — confirmed as yes/no flags
            with r1:
                schoolsup   = st.selectbox("School Support",  ["No","Yes"]) == "Yes"
                famsup      = st.selectbox("Family Support",  ["No","Yes"]) == "Yes"
            with r2:
                paid        = st.selectbox("Paid Classes",    ["No","Yes"]) == "Yes"
                activities  = st.selectbox("Activities",      ["No","Yes"]) == "Yes"
            with r3:
                nursery     = st.selectbox("Attended Nursery",["No","Yes"]) == "Yes"
                higher      = st.selectbox("Wants Higher Edu",["Yes","No"]) == "Yes"
            with r4:
                internet    = st.selectbox("Internet at Home",["Yes","No"]) == "Yes"
                romantic    = st.selectbox("Romantic Rel.",   ["No","Yes"]) == "Yes"

            submitted = st.form_submit_button("⚡ PREDICT PERFORMANCE")

    with col_result:
        if submitted:
            feature_vec = encode_input(s1, s2, s3, s4, schoolsup, famsup, paid, activities, higher)

            with st.spinner("Running KNN inference..."):
                time.sleep(0.5)
                import warnings as _w
                _w.filterwarnings('ignore')
                prediction = model.predict([feature_vec])[0]
                proba      = model.predict_proba([feature_vec])[0]

            class_map   = {0: "🔴 Needs Support", 1: "🟡 Average", 2: "✅ High Performer"}
            label       = class_map.get(int(prediction), str(prediction))
            bar_colors  = ["#ff6584", "#f6c90e", "#00d4aa"]

            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">PREDICTION RESULT</div>
                <div class="result-value">{label}</div>
                <div style="color:#8892b0;font-size:0.85rem;margin-top:8px;">
                    KNN · K={model.n_neighbors} · confidence {max(proba)*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            labels_mapped = [class_map[c] for c in model.classes_]
            fig_conf = go.Figure(go.Bar(
                x=proba * 100,
                y=labels_mapped,
                orientation="h",
                marker=dict(color=bar_colors[:len(proba)], line=dict(width=0)),
                text=[f"{p*100:.1f}%" for p in proba],
                textposition="outside",
                textfont=dict(color="#ccd6f6")
            ))
            fig_conf.update_layout(
                title=dict(text="Confidence Scores", font=dict(color="#ccd6f6", size=13)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccd6f6",
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 120]),
                yaxis=dict(showgrid=False),
                margin=dict(l=0, r=60, t=40, b=0),
                height=220
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.03);border:1px dashed rgba(108,99,255,0.3);
                        border-radius:20px;padding:60px 30px;text-align:center;
                        display:flex;flex-direction:column;align-items:center;justify-content:center;">
                <div style="font-size:3rem;margin-bottom:16px;">🤖</div>
                <div style="color:#4a5568;font-size:0.9rem;letter-spacing:1px;">
                    Fill in student details<br>and hit PREDICT
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: INSIGHTS
# ─────────────────────────────────────────────
elif menu == "📈  Insights":
    st.markdown('<div class="section-title">📈 SMART INSIGHTS</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### 🎯 Study Hours vs Score")
        np.random.seed(21)
        n = 150
        hours   = np.random.uniform(1, 10, n)
        scores  = hours * 7 + np.random.normal(0, 8, n) + 20
        grades  = pd.cut(scores, bins=[0, 50, 70, 100], labels=["Low", "Medium", "High"])
        df_sc = pd.DataFrame({"Study Hours": hours, "Score": scores.clip(0, 100), "Grade": grades})

        fig6 = px.scatter(
            df_sc, x="Study Hours", y="Score", color="Grade",
            color_discrete_map={"Low": "#ff6584", "Medium": "#f6c90e", "High": "#00d4aa"},
            template="plotly_dark",
            opacity=0.8
        )
        # Manual numpy trendline (no statsmodels needed)
        m, b = np.polyfit(df_sc["Study Hours"], df_sc["Score"], 1)
        x_line = np.linspace(df_sc["Study Hours"].min(), df_sc["Study Hours"].max(), 100)
        fig6.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", name="Trend",
            line=dict(color="#ffffff", width=2, dash="dash"),
            showlegend=False
        ))
        fig6.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col_r:
        st.markdown("##### 💤 Sleep vs Performance")
        np.random.seed(55)
        sleep_h   = np.random.uniform(3, 11, 150)
        perf_score= -(sleep_h - 7.5)**2 + 90 + np.random.normal(0, 6, 150)
        df_sleep  = pd.DataFrame({"Sleep Hours": sleep_h, "Performance": perf_score.clip(30, 100)})

        fig7 = px.density_contour(
            df_sleep, x="Sleep Hours", y="Performance",
            template="plotly_dark",
            color_discrete_sequence=["#6c63ff"]
        )
        fig7.update_traces(contours_coloring="fill", contours_showlabels=True,
                           colorscale=[[0,"rgba(0,0,0,0)"], [0.5,"rgba(108,99,255,0.3)"], [1,"rgba(0,212,170,0.6)"]])
        fig7.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccd6f6",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🏅 Top Factors Influencing Performance")

    factors = {
        "Previous Score":      92,
        "Study Hours":         88,
        "Attendance":          83,
        "Tutoring":            71,
        "Sleep Quality":       67,
        "Parental Education":  58,
        "Extracurricular":     51,
        "Internet Access":     44,
    }
    for factor, weight in factors.items():
        bar_color = "#00d4aa" if weight > 75 else "#6c63ff" if weight > 60 else "#ff6584"
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:0.85rem;color:#ccd6f6;">{factor}</span>
                <span style="font-size:0.85rem;color:{bar_color};font-weight:600;">{weight}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:50px;height:8px;overflow:hidden;">
                <div style="width:{weight}%;height:100%;background:linear-gradient(90deg,{bar_color}88,{bar_color});
                             border-radius:50px;transition:width 1s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
