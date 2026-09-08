import plotly.express as px
import streamlit as st

CHART_TEMPLATE = "plotly_dark"
CHART_COLOR_SEQUENCE = ["#2dd4bf", "#fb923c", "#60a5fa", "#f87171", "#c084fc"]

def render_chart(df, spec):
    kind = spec["kind"]
    if kind == "line":
        data = df.sort_values(spec["x"])
        fig = px.line(data, x=spec["x"], y=spec["y"], markers=True, title=spec["title"], template=CHART_TEMPLATE)
    elif kind == "bar":
        data = df.groupby(spec["x"], dropna=False)[spec["y"]].mean().reset_index()
        data = data.sort_values(spec["y"], ascending=False).head(15)
        fig = px.bar(data, x=spec["x"], y=spec["y"], title=spec["title"], template=CHART_TEMPLATE, color_discrete_sequence=CHART_COLOR_SEQUENCE)
    elif kind == "scatter":
        fig = px.scatter(df, x=spec["x"], y=spec["y"], trendline="ols", title=spec["title"], template=CHART_TEMPLATE, color_discrete_sequence=CHART_COLOR_SEQUENCE)
    elif kind == "hist":
        fig = px.histogram(df, x=spec["x"], nbins=30, title=spec["title"], template=CHART_TEMPLATE, color_discrete_sequence=CHART_COLOR_SEQUENCE)
    elif kind == "box":
        fig = px.box(df, y=spec["x"], points="outliers", title=spec["title"], template=CHART_TEMPLATE, color_discrete_sequence=CHART_COLOR_SEQUENCE)
    elif kind == "heatmap":
        corr = df[spec["columns"]].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title=spec["title"], template=CHART_TEMPLATE, color_continuous_scale="Tealgrn")
    else:
        return
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="closest",
        paper_bgcolor="#17232d",
        plot_bgcolor="#17232d",
        font={"color": "#dce8ef", "family": "Manrope, sans-serif"},
        title={"font": {"size": 17, "color": "#e5eef5"}},
        xaxis={"gridcolor": "#2e4653", "zerolinecolor": "#2e4653"},
        yaxis={"gridcolor": "#2e4653", "zerolinecolor": "#2e4653"},
    )
    st.plotly_chart(fig, width="stretch")
