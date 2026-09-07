import io
from pathlib import Path

import pandas as pd
import streamlit as st
from analyst import (
    clean_data, detect_column_types, dataset_profile, analyze_correlations,
    detect_outliers, perform_clustering, build_chart_specs, generate_insights
)
from charts import render_chart

st.set_page_config(page_title="Automated Insight Analyst", page_icon="📊", layout="wide")

def load_styles():
    styles_path = Path(__file__).parent / "styles.css"
    st.markdown(f"<style>{styles_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_styles()

st.markdown(
    """
    <section class="hero-band">
        <div class="hero-kicker">AUTOMATED ANALYTICS WORKSPACE</div>
        <h1>Insight <span>Analyst</span></h1>
        <p>Turn a raw spreadsheet into a living dashboard, clean dataset, and clear story.</p>
        <div class="hero-pills">
            <span>◉ Auto-detect</span><span>◉ Clean + profile</span><span>◉ Explore patterns</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    use_demo = st.checkbox("Use included demo dataset", value=uploaded is None)

    st.divider()
    st.write("**Pipeline**")
    st.write("1. Detect columns")
    st.write("2. Clean data")
    st.write("3. Analyze patterns")
    st.write("4. Auto-select charts")
    st.write("5. Generate insights")

@st.cache_data
def load_file(data, name):
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data))

@st.cache_data
def load_demo():
    demo_path = Path(__file__).parent / "sample_data" / "sales_demo.csv"
    return pd.read_csv(demo_path)

try:
    if uploaded:
        df_raw = load_file(uploaded.getvalue(), uploaded.name)
        source_name = uploaded.name
    elif use_demo:
        df_raw = load_demo()
        source_name = "sales_demo.csv"
    else:
        st.info("Upload a CSV/Excel file from the sidebar.")
        st.stop()

    st.markdown(
        f'<div class="dataset-strip"><span class="live-dot"></span>'
        f'<strong>{source_name}</strong><span class="strip-muted">connected</span>'
        f'<span class="strip-stat">{df_raw.shape[0]:,} source rows</span>'
        f'<span class="strip-stat">{df_raw.shape[1]:,} columns</span></div>',
        unsafe_allow_html=True,
    )

    types = detect_column_types(df_raw)

    filtered_raw = df_raw.copy()
    with st.sidebar.expander("🔎 Dashboard filters", expanded=True):
        date_columns = [column for column, kind in types.items() if kind == "date"]
        for column in date_columns[:1]:
            parsed_dates = pd.to_datetime(df_raw[column], errors="coerce", format="mixed").dropna()
            if not parsed_dates.empty:
                date_range = st.date_input(
                    f"{column} range",
                    value=(parsed_dates.min().date(), parsed_dates.max().date()),
                    key=f"filter_{column}",
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    parsed_column = pd.to_datetime(df_raw[column], errors="coerce", format="mixed")
                    start_date, end_date = map(pd.Timestamp, date_range)
                    filtered_raw = filtered_raw[parsed_column.between(start_date, end_date)]

        categorical_columns = [column for column, kind in types.items() if kind == "categorical"]
        for column in categorical_columns[:4]:
            values = sorted(df_raw[column].dropna().astype(str).unique().tolist())
            selected = st.multiselect(column, values, default=values, key=f"filter_{column}")
            filtered_raw = filtered_raw[filtered_raw[column].astype(str).isin(selected)]

    if filtered_raw.empty:
        st.warning("The selected filters contain no rows. Adjust the filters to continue.")
        st.stop()

    cleaned, cleaning_log = clean_data(filtered_raw, types)
    profile = dataset_profile(cleaned, types)
    corr = analyze_correlations(cleaned, types)
    outliers = detect_outliers(cleaned, types)
    clusters = perform_clustering(cleaned, types)
    chart_specs = build_chart_specs(cleaned, types)
    insights = generate_insights(cleaned, types, corr, outliers, clusters)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows shown", f"{len(cleaned):,}", f"of {len(df_raw):,} uploaded")
    c2.metric("Columns", f"{len(cleaned.columns):,}")
    c3.metric("Missing cells", f"{int(cleaned.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(cleaned.duplicated().sum()):,}")

    tabs = st.tabs(["📊 Dashboard", "🤖 ML & Insights", "🧹 Data Quality", "🔎 Data Preview"])

    with tabs[0]:
        st.markdown(
            '<div class="section-heading"><div><span class="eyebrow">LIVE VIEW</span>'
            '<h2>Automatically generated dashboard</h2></div>'
            '<span class="chart-count">' + str(len(chart_specs)) + ' visual signals</span></div>',
            unsafe_allow_html=True,
        )
        for spec in chart_specs:
            try:
                render_chart(cleaned, spec)
            except Exception as exc:
                st.warning(f"Could not render {spec['title']}: {exc}")

    with tabs[1]:
        st.subheader("💡 Key findings")
        if insights:
            for item in insights:
                st.markdown(f"- {item}")
        else:
            st.info("Not enough information to generate strong findings.")

        attendance_columns = [
            column for column, kind in types.items()
            if kind == "numeric" and any(token in column.lower() for token in ("attendance", "present"))
        ]
        student_columns = [
            column for column, kind in types.items()
            if kind == "categorical" and any(token in column.lower() for token in ("student", "name"))
        ]
        if attendance_columns and student_columns:
            attendance_column = attendance_columns[0]
            student_column = student_columns[0]
            st.subheader("🎓 Student attendance watchlist")
            threshold = st.slider(
                "Flag students below attendance threshold (%)",
                min_value=0,
                max_value=100,
                value=75,
                step=5,
            )
            ranking = (
                cleaned.groupby(student_column, dropna=False)[attendance_column]
                .mean()
                .sort_values(ascending=False)
                .round(2)
                .rename("Average attendance")
                .reset_index()
            )
            ranking["Status"] = ranking["Average attendance"].apply(
                lambda value: "Needs attention" if value < threshold else "On track"
            )
            flagged = ranking[ranking["Status"] == "Needs attention"]
            if flagged.empty:
                st.success(f"All students are at or above the {threshold}% threshold.")
            else:
                st.warning(f"{len(flagged)} student(s) are below the {threshold}% attendance threshold.")
                st.dataframe(flagged, use_container_width=True, hide_index=True)
            st.caption(f"Ranking based on average {attendance_column} by {student_column}.")
            st.dataframe(ranking, use_container_width=True, hide_index=True)

        st.subheader("🔗 Correlations")
        if corr.empty:
            st.info("At least two numeric columns are required for correlation analysis.")
        else:
            st.dataframe(corr, use_container_width=True, hide_index=True)

        st.subheader("🚨 Outlier detection")
        if outliers:
            for item in outliers:
                st.write(f"**{item['column']}** — {item['count']} outliers ({item['percentage']:.1f}%)")
        else:
            st.info("No suitable numeric columns found for outlier detection.")

        st.subheader("👥 Clustering")
        if clusters["available"]:
            st.write(f"Detected **{clusters['n_clusters']}** groups using K-Means.")
            st.dataframe(clusters["summary"], use_container_width=True)
        else:
            st.info(clusters["reason"])

    with tabs[2]:
        st.subheader("🧹 Automatic data cleaning")
        if cleaning_log:
            for line in cleaning_log:
                st.write("• " + line)
        else:
            st.success("No cleaning actions were necessary.")
        st.write("Detected column types:")
        st.dataframe(pd.DataFrame({
            "Column": list(types.keys()),
            "Detected Type": list(types.values())
        }), use_container_width=True, hide_index=True)

        st.subheader("Column profile")
        st.dataframe(profile, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Cleaned data")
        st.dataframe(cleaned.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download cleaned CSV",
            cleaned.to_csv(index=False).encode("utf-8"),
            "cleaned_dataset.csv",
            "text/csv"
        )

except Exception as exc:
    st.error("The dataset could not be processed.")
    st.exception(exc)
