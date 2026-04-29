import dash
from dash import html, dcc, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go

dash.register_page(__name__, path="/distributions")

# Using state-day outliers dataset gives us per-day, per-state points (great for distributions)
df = pd.read_pickle("data/processed/outliers_state_day.pkl")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

METRIC_LABELS = {
    "total_damage_usd": "Total Damage (USD)",
    "fatalities": "Fatalities",
    "storm_count": "Storm Count",
    "severity": "Severity Score",
}

def make_ccdf(values, title, x_label):
    x = np.array(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    x.sort()

    # Complementary cumulative share: how many extreme state-days meet or exceed each value.
    n = len(x)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title="No data to plot")
        return fig

    y = 1.0 - (np.arange(1, n + 1) / n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Extreme-state-day curve"))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Share of extreme state-days at or above this value",
        yaxis=dict(tickformat=".2f"),
        xaxis=dict(type="log"),  # log x-scale is standard for heavy tails
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig

layout = html.Div([
    html.Div(className="card", children=[
        html.H3("Chapter 3 — Distributions & extremes"),
        html.P(
            "This chart highlights how quickly extreme storm impacts taper off. "
            "The x-axis uses a log scale because storm impacts are heavily skewed.",
            className="small",
        ),

        html.Div(className="controls", children=[
            html.Div(className="control", children=[
                html.Label("Metric", className="small"),
                dcc.Dropdown(
                    id="ccdf_metric",
                    options=[
                        {"label": "Total Damage (USD)", "value": "total_damage_usd"},
                        {"label": "Fatalities", "value": "fatalities"},
                        {"label": "Storm Count", "value": "storm_count"},
                        {"label": "Severity Score", "value": "severity"},
                    ],
                    value="total_damage_usd",
                    clearable=False,
                ),
            ])
        ]),

        dcc.Graph(id="ccdf_plot", style={"height": "520px"}, config={"displayModeBar": False}),
    ])
])

@dash.callback(
    Output("ccdf_plot", "figure"),
    Input("ccdf_metric", "value"),
)
def update_ccdf(metric):
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    title = f"Extreme-state-day distribution for {metric_label} (2024)"
    return make_ccdf(df[metric].fillna(0), title, f"{metric_label} per state-day (log scale)")
