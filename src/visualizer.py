import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#1B2D45",
        font=dict(color="#E0E0E0"),
        title=dict(font=dict(size=18, color="#00B4D8")),
        xaxis=dict(gridcolor="#2A3F5F", zerolinecolor="#2A3F5F"),
        yaxis=dict(gridcolor="#2A3F5F", zerolinecolor="#2A3F5F"),
        legend=dict(font=dict(color="#E0E0E0")),
        hoverlabel=dict(bgcolor="#1B2D45", font_size=12, font_color="#FFFFFF"),
    )
)

def _apply_dark(fig):
    fig.update_layout(
        paper_bgcolor="#0D1B2A", plot_bgcolor="#1B2D45",
        font=dict(color="#E0E0E0"),
        title=dict(font=dict(size=18, color="#00B4D8")),
        xaxis=dict(gridcolor="#2A3F5F", zerolinecolor="#2A3F5F"),
        yaxis=dict(gridcolor="#2A3F5F", zerolinecolor="#2A3F5F"),
        hoverlabel=dict(bgcolor="#1B2D45", font_size=12, font_color="#FFFFFF"),
    )
    return fig

def line_chart_units(df: pd.DataFrame):
    df = df.copy().sort_values(["year", "month"])
    df["label"] = df["month"].apply(lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]) + \
                  " " + df["year"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["units_consumed"],
        mode="lines+markers", name="Units Consumed",
        line=dict(color="#00B4D8", width=3),
        marker=dict(size=8, color="#00B4D8")
    ))
    if len(df) >= 3:
        z = np.polyfit(range(len(df)), df["units_consumed"], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df["label"], y=p(range(len(df))),
            mode="lines", name="Trend",
            line=dict(color="#FFB703", width=2, dash="dash")
        ))
    fig.update_layout(title="Monthly Consumption (kWh)", hovermode="x unified")
    return _apply_dark(fig)

def bar_chart_cost(df: pd.DataFrame):
    df = df.copy().sort_values(["year", "month"])
    df["label"] = df["month"].apply(lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]) + \
                  " " + df["year"].astype(str)
    df["change"] = df["net_payable"].diff()
    colors = ["#00B4D8" if i == 0 else ("#00E676" if c <= 0 else "#FF5252")
              for i, c in enumerate(df["change"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["label"], y=df["net_payable"],
        marker_color=colors,
        text=[f"₹{v:,.0f}" for v in df["net_payable"]],
        textposition="outside", textfont=dict(size=10)
    ))
    fig.update_layout(title="Monthly Bill Amount (₹)", hovermode="x unified")
    _apply_dark(fig)
    fig.update_yaxes(tickprefix="₹")
    return fig

def stacked_cost_breakdown(df: pd.DataFrame):
    df = df.copy().sort_values(["year", "month"])
    df["label"] = df["month"].apply(lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]) + \
                  " " + df["year"].astype(str)
    fig = go.Figure()
    categories = [
        ("fixed_charges", "#4A90D9"), ("energy_charges", "#00B4D8"),
        ("fppca_charges", "#7BDFF2"), ("pg_surcharge", "#FFB703"),
        ("tax_amount", "#FF8C00"), ("md_penalty", "#FF5252"),
        ("true_up_charges", "#9C27B0"), ("arrears", "#E91E63")
    ]
    for col, color in categories:
        if col in df.columns and df[col].sum() > 0:
            fig.add_trace(go.Bar(
                name=col.replace("_", " ").title(),
                x=df["label"], y=df[col],
                marker_color=color
            ))
    fig.update_layout(barmode="stack", title="Monthly Cost Breakdown")
    _apply_dark(fig)
    return fig

def area_chart_cumulative(df: pd.DataFrame):
    df = df.copy().sort_values(["year", "month"])
    df["label"] = df["month"].apply(lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]) + \
                  " " + df["year"].astype(str)
    df["cumulative"] = df["net_payable"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["cumulative"],
        fill="tozeroy", mode="lines+markers",
        line=dict(color="#00B4D8", width=3),
        fillcolor="rgba(0, 180, 216, 0.2)"
    ))
    fig.update_layout(title="Cumulative Annual Spend (₹)", hovermode="x unified")
    _apply_dark(fig)
    return fig

def heatmap_consumption(df: pd.DataFrame):
    if df.empty:
        return None
    df = df.copy()
    pivot = df.pivot_table(index="year", columns="month", values="units_consumed", aggfunc="mean")
    pivot.columns = [["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"][c-1] for c in pivot.columns]
    fig = px.imshow(pivot, text_auto=".0f", aspect="auto",
                    title="Consumption Heatmap (kWh)",
                    labels={"x": "Month", "y": "Year", "color": "kWh"},
                    color_continuous_scale="Blues")
    fig.update_layout(title=dict(font=dict(color="#00B4D8")))
    _apply_dark(fig)
    return fig

def forecast_chart(forecast_df: pd.DataFrame, history_df: pd.DataFrame):
    history_df = history_df.copy().sort_values(["year", "month"])
    history_df["label"] = history_df["month"].apply(
        lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]) + " " + \
                   history_df["year"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["label"], y=history_df["units_consumed"],
        mode="lines+markers", name="Historical",
        line=dict(color="#00B4D8", width=3),
        marker=dict(size=8)
    ))
    if "ds" in forecast_df.columns:
        forecast_df = forecast_df.copy()
        forecast_df["label"] = forecast_df["month"].apply(
            lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]) + " " + \
                       forecast_df["year"].astype(str)
        fig.add_trace(go.Scatter(
            x=forecast_df["label"], y=forecast_df["yhat"],
            mode="lines+markers", name="Forecast",
            line=dict(dash="dash", color="#FFB703", width=3),
            marker=dict(size=8, color="#FFB703")
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["label"], y=forecast_df["yhat_upper"],
            fill=None, mode="lines", line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["label"], y=forecast_df["yhat_lower"],
            fill="tonexty", mode="lines", line=dict(width=0),
            name="Confidence Band",
            fillcolor="rgba(255, 183, 3, 0.15)"
        ))
    fig.update_layout(title="Units Consumption Forecast", hovermode="x unified")
    _apply_dark(fig)
    return fig
