"""AI-Based Student Performance Analysis System
Developed by Mohamed Faisal

Start the dashboard with:
    streamlit run app.py
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets Python find the src/ folder

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config as cfg
from src import visuals as vz
from src.data_processing import (
    REQUIRED_COLUMNS,
    category_counts,
    clean_data,
    generate_insights,
    get_suggestions,
    summary_statistics,
    validate_columns,
)
from src.model import predict_one, train_models

st.set_page_config(page_title=cfg.APP_TITLE, page_icon="🎓", layout="wide")

PAGES = [
    "📊 Dashboard",
    "👤 Student Profile",
    "🔮 Predict Performance",
    "🧪 Model Performance",
    "🗂️ Data & Cleaning",
    "ℹ️ About",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
p, li, label, h1, h2, h3, h4, .hero, .kpi, .footer, .about-row, .chip, .badge, .side-brand {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1280px;}
section[data-testid="stSidebar"] {background: #f8fafc;}

.hero {
    background: linear-gradient(120deg, #1e1b4b 0%, #4338ca 55%, #0ea5e9 100%);
    padding: 26px 32px; border-radius: 16px; margin-bottom: 1.2rem;
    box-shadow: 0 10px 28px rgba(67, 56, 202, 0.25);
}
.hero-title {color: #ffffff; font-size: 1.85rem; font-weight: 700; line-height: 1.25;}
.hero-sub {color: #c7d2fe; font-size: 1.02rem; font-weight: 500; margin-top: 6px;}

.kpi {
    background: #ffffff; border: 1px solid #e5e7eb; border-left: 5px solid var(--accent, #4f46e5);
    border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}
.kpi .label {color: #6b7280; font-size: 0.74rem; text-transform: uppercase;
             letter-spacing: 0.06em; font-weight: 600;}
.kpi .value {color: #111827; font-size: 1.7rem; font-weight: 700; line-height: 1.25;}
.kpi .sub {color: #6b7280; font-size: 0.8rem;}

.badge {color: #ffffff; padding: 3px 12px; border-radius: 999px; font-weight: 600; font-size: 0.95rem;}
.chip {display: inline-block; background: #eef2ff; color: #4338ca; padding: 5px 14px;
       border-radius: 999px; font-size: 0.85rem; font-weight: 600; margin: 0 6px 6px 0;}

.about-row {display: flex; padding: 11px 0; border-bottom: 1px solid #eef0f3; font-size: 0.97rem;}
.about-row .k {width: 160px; color: #6b7280; font-weight: 600;}
.about-row .v {color: #111827; font-weight: 500; flex: 1;}

.credit {text-align: center; color: #6b7280; font-size: 0.88rem; margin-top: 10px;}
.credit b {color: #4338ca;}
.side-brand {padding: 4px 2px 10px 2px; font-size: 1.05rem; color: #111827;}
.side-brand span {color: #6b7280; font-size: 0.82rem;}
.footer {text-align: center; color: #6b7280; font-size: 0.86rem; padding: 18px 0 6px 0;
         border-top: 1px solid #e5e7eb; margin-top: 2.2rem;}
</style>
"""


# ------------------------------------------------------------ UI helpers
def render_header():
    st.markdown(
        f'<div class="hero"><div class="hero-title">🎓 {cfg.APP_TITLE}</div>'
        f'<div class="hero-sub">{cfg.SUBTITLE}</div></div>',
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(f'<div class="footer">{cfg.FOOTER}</div>', unsafe_allow_html=True)


def kpi(label, value, sub="", accent="#4f46e5"):
    st.markdown(
        f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def badge(category):
    color = cfg.CATEGORY_COLORS.get(category, "#6b7280")
    return f'<span class="badge" style="background:{color}">{category}</span>'


def show(fig):
    """Display a Matplotlib figure and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------ data access
@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes):
    """Read the CSV (default file or an uploaded one) and clean it."""
    if file_bytes is None:
        raw = pd.read_csv(cfg.DATA_PATH)
    else:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    missing_columns = validate_columns(raw)
    if missing_columns:
        return raw, None, None, missing_columns
    clean, report = clean_data(raw)
    return raw, clean, report, []


@st.cache_resource(show_spinner="Training the AI model...")
def load_model(clean):
    return train_models(clean)


# ------------------------------------------------------------- login page
def login_page():
    render_header()
    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        with st.container(border=True):
            st.subheader("🔐 Sign in")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
            st.caption(
                f"Demo login → username: **{cfg.DEMO_USERNAME}**  |  password: **{cfg.DEMO_PASSWORD}**"
            )
            st.markdown(
                f'<div class="credit">Created by <b>{cfg.DEVELOPER}</b> · {cfg.DEPARTMENT}</div>',
                unsafe_allow_html=True,
            )
        if submitted:
            if username == cfg.DEMO_USERNAME and password == cfg.DEMO_PASSWORD:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    render_footer()


# ---------------------------------------------------------------- filters
def sidebar_filters(df):
    st.markdown("### 🔎 Filters")
    categories = st.multiselect("Performance category", cfg.CATEGORY_ORDER, default=cfg.CATEGORY_ORDER)
    attendance = st.slider("Attendance (%)", 0, 100, (0, 100))
    picked = st.multiselect("Select students", df["Label"].tolist(), placeholder="All students")

    view = df[df["Category"].isin(categories) & df["Attendance"].between(attendance[0], attendance[1])]
    if picked:
        view = view[view["Label"].isin(picked)]
    st.caption(f"Showing {len(view)} of {len(df)} students")
    return view


# -------------------------------------------------------------- dashboard
def page_dashboard(df, view, bundle):
    if view.empty:
        st.warning("No students match the current filters. Please change the filters in the sidebar.")
        return

    metrics = bundle["metrics"]
    cols = st.columns(5)
    with cols[0]:
        kpi("Total Students", f"{len(view):,}", f"of {len(df):,} in the dataset", "#4f46e5")
    with cols[1]:
        kpi("Average Marks", f"{view[cfg.TARGET].mean():.1f}", "final exam, out of 100", "#10b981")
    with cols[2]:
        kpi("Average Attendance", f"{view['Attendance'].mean():.1f}%", "across selected students", "#3b82f6")
    with cols[3]:
        kpi("Avg Study Hours", f"{view['Study_Hours'].mean():.1f}", "hours per day", "#f59e0b")
    with cols[4]:
        kpi("Model Accuracy", f"{metrics['cv_accuracy']:.0%}", "5-fold cross-validation", "#8b5cf6")

    st.write("")
    with st.expander("🧠 Key insights from the data", expanded=True):
        for note in generate_insights(view):
            st.markdown(f"- {note}")

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            mode = st.radio("Show", ["Top performers", "Needs attention"], horizontal=True)
            max_n = min(25, len(view))
            n = st.slider("Number of students", 3, max_n, min(10, max_n)) if max_n > 3 else max_n
            show(vz.bar_student_marks(view, n, top=(mode == "Top performers")))
    with right:
        with st.container(border=True):
            show(vz.pie_categories(view))

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            show(vz.line_class_progress(view))
    with right:
        with st.container(border=True):
            show(vz.scatter_vs_marks(view, "Attendance", "Attendance (%)", "Attendance vs final marks"))

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            show(vz.heatmap_correlation(view))
    with right:
        with st.container(border=True):
            show(vz.scatter_vs_marks(view, "Study_Hours", "Study hours per day", "Study hours vs final marks"))

    st.subheader("🤖 Student results and AI predictions")
    tab_pred, tab_high, tab_low, tab_stats = st.tabs(
        ["AI predictions", "High performers", "Low performers", "Summary statistics"]
    )
    with tab_pred:
        table = view[
            ["Student_ID", "Name", "Attendance", "Study_Hours", "Internal_Marks", "Assignment_Marks",
             "Previous_Sem_Marks", "Final_Marks", "Category", "Predicted_Marks", "Predicted_Category"]
        ].copy()
        table["Match"] = (view["Category"] == view["Predicted_Category"]).map({True: "✅", False: "⚠️"})
        table.columns = ["Student ID", "Name", "Attendance %", "Study Hrs", "Internal", "Assignment",
                         "Prev Sem", "Final Marks", "Actual Category", "AI Marks", "AI Category", "Match"]
        st.dataframe(table, hide_index=True)
        st.caption(
            "Each prediction comes from a model that never saw that student during training "
            "(cross-validation), so the Match column is an honest test of the AI."
        )
    with tab_high:
        high = view[view["Category"] == "Excellent"].sort_values(cfg.TARGET, ascending=False)
        st.caption(f"{len(high)} students scored 80 or above.")
        st.dataframe(
            high[["Student_ID", "Name", "Attendance", "Study_Hours", cfg.TARGET]], hide_index=True
        )
    with tab_low:
        low = view[view["Category"] == "Needs Improvement"].sort_values(cfg.TARGET)
        st.caption(f"{len(low)} students scored below 55 and need extra support.")
        st.dataframe(
            low[["Student_ID", "Name", "Attendance", "Study_Hours", cfg.TARGET]], hide_index=True
        )
    with tab_stats:
        st.dataframe(summary_statistics(view))


# ---------------------------------------------------------- student page
def page_student(df, bundle):
    st.subheader("👤 Individual student profile")
    choice = st.selectbox("Select a student", df["Label"].tolist())
    row = df[df["Label"] == choice].iloc[0]

    cols = st.columns(4)
    with cols[0]:
        kpi("Attendance", f"{row['Attendance']:.1f}%", "", "#3b82f6")
    with cols[1]:
        kpi("Internal Marks", f"{row['Internal_Marks']:.1f}", "out of 100", "#4f46e5")
    with cols[2]:
        kpi("Assignment Marks", f"{row['Assignment_Marks']:.1f}", "out of 100", "#10b981")
    with cols[3]:
        kpi("Study Hours", f"{row['Study_Hours']:.1f}", "per day", "#f59e0b")

    st.write("")
    cols = st.columns(3)
    with cols[0]:
        kpi("Actual Result", badge(row["Category"]), f"Final marks: {row[cfg.TARGET]:.1f}", "#64748b")
    with cols[1]:
        kpi("AI Predicted Category", badge(row["Predicted_Category"]),
            "predicted from the 5 input features", "#4f46e5")
    with cols[2]:
        kpi("AI Predicted Marks", f"{row['Predicted_Marks']:.1f}",
            f"actual {row[cfg.TARGET]:.1f}", "#8b5cf6")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            show(vz.line_student_progress(row, df))
    with right:
        with st.container(border=True):
            show(vz.bar_student_vs_class(row, df))

    st.subheader("💡 Personalised suggestions")
    for tip in get_suggestions({f: row[f] for f in cfg.FEATURES}):
        st.markdown(f"- {tip}")


# ------------------------------------------------------- prediction page
def page_predict(bundle):
    st.subheader("🔮 Predict a student's performance")
    st.caption("Move the sliders and the AI updates its prediction instantly.")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            attendance = st.slider("Attendance (%)", 0, 100, 75)
            internal = st.slider("Internal exam marks", 0, 100, 65)
            assignment = st.slider("Assignment marks", 0, 100, 70)
        with right:
            previous = st.slider("Previous semester marks", 0, 100, 65)
            study = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)

    values = {
        "Attendance": attendance,
        "Internal_Marks": internal,
        "Assignment_Marks": assignment,
        "Previous_Sem_Marks": previous,
        "Study_Hours": study,
    }
    category, probabilities, marks = predict_one(bundle, values)

    cols = st.columns(3)
    with cols[0]:
        kpi("Predicted Category", badge(category), "", cfg.CATEGORY_COLORS[category])
    with cols[1]:
        kpi("Predicted Final Marks", f"{marks:.1f}", "out of 100", "#4f46e5")
    with cols[2]:
        kpi("Model Confidence", f"{probabilities[category]:.0%}", "probability of this category", "#8b5cf6")

    st.write("")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            show(vz.probability_chart(probabilities))
    with right:
        with st.container(border=True):
            st.markdown("#### 💡 Suggestions")
            for tip in get_suggestions(values):
                st.markdown(f"- {tip}")


# ------------------------------------------------------------ model page
def page_model(bundle):
    st.subheader("🧪 Model performance")
    m = bundle["metrics"]

    cols = st.columns(4)
    with cols[0]:
        kpi("Accuracy", f"{m['accuracy']:.1%}", f"on {m['test_size']} unseen test students", "#4f46e5")
    with cols[1]:
        kpi("Precision", f"{m['precision']:.1%}", "weighted average", "#10b981")
    with cols[2]:
        kpi("Recall", f"{m['recall']:.1%}", "weighted average", "#3b82f6")
    with cols[3]:
        kpi("F1-Score", f"{m['f1']:.1%}", "weighted average", "#f59e0b")
    st.write("")
    cols = st.columns(3)
    with cols[0]:
        kpi("Cross-validated accuracy", f"{m['cv_accuracy']:.1%}", f"± {m['cv_std']:.1%} (5-fold)", "#8b5cf6")
    with cols[1]:
        kpi("R² score (marks)", f"{m['r2']:.2f}", "1.00 would be perfect", "#0ea5e9")
    with cols[2]:
        kpi("Mean absolute error", f"{m['mae']:.1f} marks", "average prediction error", "#ef4444")

    st.info(
        f"The test set has only {m['test_size']} students, so a single accuracy number can be lucky or "
        "unlucky. The cross-validated accuracy tests the model 5 times and is a steadier estimate."
    )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            show(vz.feature_importance_chart(bundle["importance"]))
    with right:
        with st.container(border=True):
            show(vz.confusion_matrix_chart(bundle["confusion_matrix"], bundle["labels"]))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Model comparison")
        st.dataframe(bundle["comparison"], hide_index=True)
    with right:
        st.markdown("#### Classification report")
        st.dataframe(bundle["report_df"])

    with st.expander("📘 How does the Random Forest algorithm work? (simple explanation)"):
        st.markdown(
            """
- A **decision tree** asks simple yes/no questions, such as *"Is attendance above 75%?"*, and ends at a prediction.
- One tree can be wrong, so a **Random Forest** builds **200 different trees**. Each tree sees a slightly different
  random sample of students.
- To predict a new student, every tree **votes** and the most popular category wins. Many opinions together are
  more reliable than a single one.
- The forest also shows **which inputs mattered most** (the feature importance chart).
- Two forests are used: a **classifier** (predicts the category) and a **regressor** (predicts the marks).
            """
        )


# --------------------------------------------------------------- data page
def page_data(raw, clean, report):
    st.subheader("🗂️ Data and cleaning")
    cols = st.columns(4)
    with cols[0]:
        kpi("Rows in raw file", f"{report['rows_before']}", "", "#4f46e5")
    with cols[1]:
        kpi("Duplicates removed", f"{report['duplicates_removed']}", "same student ID twice", "#f59e0b")
    with cols[2]:
        kpi("Missing values", f"{sum(report['missing_found'].values())}", "filled with the median", "#3b82f6")
    with cols[3]:
        kpi("Impossible values", f"{sum(report['invalid_found'].values())}", "e.g. attendance above 100%", "#ef4444")

    st.write("")
    tab_clean, tab_raw, tab_report, tab_stats = st.tabs(
        ["Cleaned data", "Raw data", "Cleaning report", "Summary statistics"]
    )
    with tab_clean:
        st.dataframe(clean, hide_index=True)
        st.download_button(
            "⬇️ Download cleaned data (CSV)",
            clean.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_students.csv",
            mime="text/csv",
        )
    with tab_raw:
        st.caption("The file exactly as it was loaded, problems included.")
        st.dataframe(raw, hide_index=True)
    with tab_report:
        rows = [
            ("Rows in raw file", str(report["rows_before"]), "Starting point"),
            ("Duplicate students removed", str(report["duplicates_removed"]), "Kept the first row for each Student ID"),
            ("Missing values", str(sum(report["missing_found"].values())), str(report["missing_found"] or "none")),
            ("Impossible values", str(sum(report["invalid_found"].values())), str(report["invalid_found"] or "none")),
            ("Rows without final marks dropped", str(report["rows_dropped_no_final"]), "Cannot learn from them"),
            ("Rows after cleaning", str(report["rows_after"]), "Used for analysis and AI"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Step", "Count", "Details"]), hide_index=True)
        st.markdown(
            "**How problems are handled:** names are tidied; duplicates removed; impossible values "
            "(for example attendance 112% or study hours below 0) are treated as missing; "
            "missing values are filled with the column **median**."
        )
    with tab_stats:
        st.dataframe(summary_statistics(clean))
        st.markdown("**Students per category**")
        st.dataframe(category_counts(clean).rename("Students").to_frame())


# --------------------------------------------------------------- about page
def page_about():
    st.subheader("ℹ️ About this project")
    technologies = cfg.TECHNOLOGIES
    tech_text = ", ".join(technologies[:-1]) + ", and " + technologies[-1]

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("#### 👨‍💻 Developer details")
            st.markdown(
                f'<div class="about-row"><div class="k">Developer</div><div class="v">{cfg.DEVELOPER}</div></div>'
                f'<div class="about-row"><div class="k">Department</div><div class="v">{cfg.DEPARTMENT}</div></div>'
                f'<div class="about-row"><div class="k">Project</div><div class="v">{cfg.APP_TITLE}</div></div>'
                f'<div class="about-row"><div class="k">Technologies</div><div class="v">{tech_text}</div></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            chips = "".join(f'<span class="chip">{t}</span>' for t in technologies)
            st.markdown(chips, unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown("#### 🎯 What it does")
            st.markdown(
                "- Cleans and analyses student data\n"
                "- Finds high and low performers\n"
                "- Predicts performance with Machine Learning\n"
                "- Shows what influences marks the most\n"
                "- Gives personal improvement tips"
            )

    with st.container(border=True):
        st.markdown("#### ⚙️ How it works")
        steps = st.columns(4)
        texts = [
            ("1 · Load & Clean", "CSV data is loaded; duplicates, gaps and impossible values are fixed."),
            ("2 · Analyse", "Statistics, correlations and insights are calculated with Pandas."),
            ("3 · Learn", "Random Forest models learn the link between habits and results."),
            ("4 · Predict", "The dashboard predicts category and marks for every student."),
        ]
        for col, (title, body) in zip(steps, texts):
            with col:
                st.markdown(f"**{title}**")
                st.caption(body)

    st.caption(
        "Data note: the included dataset is synthetic (made-up) and only for demonstration. "
        "Use the sidebar upload to analyse your own CSV file."
    )


# ------------------------------------------------------------------- main
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    if not st.session_state.get("logged_in", False):
        login_page()
        return

    # ---- sidebar
    st.sidebar.markdown(
        f'<div class="side-brand">🎓 <b>Student Performance AI</b><br>'
        f"<span>by {cfg.DEVELOPER}</span></div>",
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    filter_box = st.sidebar.container()
    with st.sidebar.expander("📂 Use your own data"):
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    render_header()

    # ---- data and model
    try:
        raw, clean, report, missing_columns = load_and_clean(uploaded.getvalue() if uploaded else None)
    except FileNotFoundError:
        st.error("data/students.csv was not found. Run `python data/generate_data.py` first.")
        render_footer()
        return
    except Exception as error:  # a bad upload, for example
        st.error(f"Could not read the data file: {error}")
        render_footer()
        return

    if missing_columns:
        st.error("The uploaded file is missing these columns: " + ", ".join(missing_columns))
        render_footer()
        return

    try:
        bundle = load_model(clean)
    except ValueError as error:
        st.error(str(error))
        render_footer()
        return
    scored = bundle["scored"]

    # ---- pages
    if page == PAGES[0]:
        with filter_box:
            view = sidebar_filters(scored)
        page_dashboard(scored, view, bundle)
    elif page == PAGES[1]:
        page_student(scored, bundle)
    elif page == PAGES[2]:
        page_predict(bundle)
    elif page == PAGES[3]:
        page_model(bundle)
    elif page == PAGES[4]:
        page_data(raw, clean, report)
    else:
        page_about()

    render_footer()


main()
