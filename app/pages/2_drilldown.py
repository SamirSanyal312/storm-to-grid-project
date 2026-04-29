# app/pages/2_drilldown.py
import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

dash.register_page(__name__, path="/drilldown")

storms_region = pd.read_pickle("data/processed/storms_region_daily.pkl")
eia_custom = pd.read_pickle("data/processed/eia_custom_region_daily.pkl")

storms_region["date"] = pd.to_datetime(storms_region["date"])
eia_custom["date"] = pd.to_datetime(eia_custom["date"])

CUSTOM_REGIONS = sorted(storms_region["region"].dropna().unique().tolist())

STORM_METRIC_LABELS = {
    "storm_count": "Storm Count",
    "severity": "Severity Score",
    "fatalities": "Fatalities",
    "total_damage_usd": "Total Damage (USD)",
}

# (label -> (column in eia_custom, axis title))
STRESS_OPTIONS = {
    "Typical Grid Forecast Miss": ("anomaly_abs_mean_mw", "Average Demand Forecast Miss (MW)"),
    "Largest Grid Forecast Miss": ("anomaly_abs_max_mw", "Largest Demand Forecast Miss (MW)"),
    "Average Demand vs Forecast": ("anomaly_mean_mw", "Average Demand Minus Forecast (MW)"),
}


# -------------------------
# Spike markers (keep as-is)
# -------------------------
def add_spike_markers(fig: go.Figure, region: str, which: list[str], top_n: int = 5):
    d = storms_region[storms_region["region"] == region].sort_values("date")

    # Count spikes
    if "count" in which:
        top = d.nlargest(top_n, "storm_count")[["date", "storm_count"]]
        for _, r in top.iterrows():
            fig.add_vline(
                x=r["date"],
                line_width=1,
                line_dash="dash",
                line_color="rgba(56,189,248,0.55)",
            )
            fig.add_annotation(
                x=r["date"], y=1.02, xref="x", yref="paper",
                text=f"Count: {int(r['storm_count'])}",
                showarrow=False,
                font=dict(size=10, color="rgba(56,189,248,0.95)"),
            )

    # Severity spikes
    if "severity" in which:
        top = d.nlargest(top_n, "severity")[["date", "severity"]]
        for _, r in top.iterrows():
            fig.add_vline(
                x=r["date"],
                line_width=1,
                line_dash="dot",
                line_color="rgba(251,191,36,0.55)",
            )
            fig.add_annotation(
                x=r["date"], y=0.96, xref="x", yref="paper",
                text=f"Sev: {r['severity']:.0f}",
                showarrow=False,
                font=dict(size=10, color="rgba(251,191,36,0.95)"),
            )


# -------------------------
# Main timeline (keep as-is)
# -------------------------
def make_timeline(region: str, storm_metric: str, stress_label: str, spike_markers: list[str]):
    stress_col, y2_title = STRESS_OPTIONS[stress_label]

    s = storms_region[storms_region["region"] == region].sort_values("date")
    e = (
        eia_custom[eia_custom["custom_region"] == region][["date", stress_col]]
        .sort_values("date")
        .rename(columns={stress_col: "grid_value"})
    )

    merged = pd.merge(
        s[["date", storm_metric]].rename(columns={storm_metric: "storm_value"}),
        e,
        on="date",
        how="left",
    )

    storm_legend_name = STORM_METRIC_LABELS.get(storm_metric, storm_metric)

    fig = go.Figure()

    # Storm (left axis)
    fig.add_trace(go.Scatter(
        x=merged["date"],
        y=merged["storm_value"],
        name=storm_legend_name,
        mode="lines",
        line=dict(width=2),
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            f"{STORM_METRIC_LABELS.get(storm_metric, storm_metric)}: <b>%{{y:,.2f}}</b>"
            "<extra></extra>"
        ),
    ))

    # Grid (right axis)
    fig.add_trace(go.Scatter(
        x=merged["date"],
        y=merged["grid_value"],
        name=stress_label,
        mode="lines",
        line=dict(width=2, dash="dot"),
        yaxis="y2",
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            f"{stress_label}: <b>%{{y:,.0f}}</b>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"Storm → Grid (2024) — Region: {region}",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            font=dict(size=18, color="#e5e7eb"),
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=120, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        xaxis=dict(
            title="Date",
            showgrid=False,
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
            title=STORM_METRIC_LABELS.get(storm_metric, storm_metric),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        yaxis2=dict(
            title=y2_title,
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.35)",
            zerolinewidth=2,
        ),
    )

    spike_markers = spike_markers or []
    if spike_markers:
        add_spike_markers(fig, region, spike_markers, top_n=5)

    return fig


# -------------------------
# KPI + Sparklines (already in your file)
# -------------------------
def _sparkline_figure(dates: pd.Series, values: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode="lines",
        line=dict(width=2),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.update_layout(
        height=56,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def kpis_for_region(region: str):
    s = storms_region[storms_region["region"] == region].sort_values("date")
    return {
        "storms_total": int(s["storm_count"].sum()),
        "fatalities_total": int(s["fatalities"].sum()),
        "damage_total": float(s["total_damage_usd"].sum()),
        "storms_series": (s["date"], s["storm_count"].fillna(0)),
        "fatalities_series": (s["date"], s["fatalities"].fillna(0)),
        "damage_series": (s["date"], s["total_damage_usd"].fillna(0)),
    }


def kpi_card(title: str, value_text: str, spark_fig: go.Figure, accent_class: str = ""):
    return html.Div(
        className=f"kpi {accent_class}".strip(),
        children=[
            html.Div(title, className="label"),
            html.Div(value_text, className="value"),
            dcc.Graph(
                figure=spark_fig,
                className="sparkline",
                config={"displayModeBar": False},
            ),
        ],
    )


# -------------------------
# NEW: Small multiples (Regional Patterns)
# -------------------------
def _month_label(dt: pd.Timestamp) -> str:
    return dt.strftime("%b")

def make_regional_small_multiples(stress_label: str) -> go.Figure:
    """
    Matches approved image:
      - storms: solid gold
      - grid: dashed cyan
      - monthly aggregation
      - small multiples by region
      - single legend at top
    """
    stress_col, _y2_title = STRESS_OPTIONS[stress_label]

    # Monthly storms (sum)
    s = storms_region.copy()
    s["month"] = s["date"].dt.to_period("M").dt.to_timestamp()
    s_month = (
        s.groupby(["region", "month"], as_index=False)
        .agg(storms=("storm_count", "sum"))
    )

    # Monthly grid (mean) — forecast miss style metric makes most sense as mean level
    e = eia_custom.copy()
    e["month"] = e["date"].dt.to_period("M").dt.to_timestamp()
    e_month = (
        e.groupby(["custom_region", "month"], as_index=False)
        .agg(grid=(stress_col, "mean"))
        .rename(columns={"custom_region": "region"})
    )

    m = pd.merge(s_month, e_month, on=["region", "month"], how="left")
    regions = [r for r in CUSTOM_REGIONS if r in m["region"].unique()]

    # Layout like the screenshot: 4 columns grid
    cols = 4
    rows = max(1, math.ceil(len(regions) / cols))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=regions,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.07,
    )

    # Colors to match screenshot
    storms_color = "#fbbf24"   # gold
    grid_color = "#22d3ee"     # cyan

    # Show legend only once (first subplot)
    legend_done = False

    for i, region in enumerate(regions):
        r = (i // cols) + 1
        c = (i % cols) + 1

        d = m[m["region"] == region].sort_values("month")
        # Ensure Jan..Dec ordering if present
        # (Your dataset is 2024-only, so this will align naturally.)

        fig.add_trace(
            go.Scatter(
                x=d["month"],
                y=d["storms"],
                mode="lines",
                line=dict(width=2.5, color=storms_color),
                name="Storms",
                showlegend=(not legend_done),
                hovertemplate="<b>%{x|%b %Y}</b><br>Storms: <b>%{y:,.0f}</b><extra></extra>",
            ),
            row=r, col=c
        )

        fig.add_trace(
            go.Scatter(
                x=d["month"],
                y=d["grid"],
                mode="lines",
                line=dict(width=2.5, color=grid_color, dash="dash"),
                name="Grid Anomaly",
                showlegend=(not legend_done),
                hovertemplate="<b>%{x|%b %Y}</b><br>Grid anomaly: <b>%{y:,.0f}</b> MW<extra></extra>",
            ),
            row=r, col=c
        )

        legend_done = True

        # style axes per subplot
        fig.update_xaxes(
            row=r, col=c,
            tickformat="%b",
            showgrid=False,
            ticks="outside",
        )
        fig.update_yaxes(
            row=r, col=c,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        )

    fig.update_layout(
        title=dict(
            text="Regional Patterns",
            x=0.01,
            xanchor="left",
            font=dict(size=22, color="#e5e7eb"),
        ),
        margin=dict(l=40, r=40, t=90, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.08,
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig


# -------------------------
# Layout
# -------------------------
layout = html.Div([
    html.Div(className="card", children=[
        html.H3("Chapter 2 — Drilldown: Storm → Grid timeline (aligned regions)"),
        html.P(
            "Compare storm activity against grid behavior within the same custom region. "
            "Switch the grid metric between typical forecast miss, largest forecast miss, "
            "or whether demand usually ran above or below forecast. Toggle spike markers for storytelling.",
            className="small",
        ),

        html.Div(className="controls", children=[
            html.Div(className="control", children=[
                html.Label("Custom region", className="small"),
                dcc.Dropdown(
                    id="custom_region",
                    options=[{"label": r, "value": r} for r in CUSTOM_REGIONS],
                    value=CUSTOM_REGIONS[0] if CUSTOM_REGIONS else None,
                    clearable=False,
                ),
            ]),
            html.Div(className="control", children=[
                html.Label("Storm metric", className="small"),
                dcc.Dropdown(
                    id="storm_metric",
                    options=[
                        {"label": "Storm Count", "value": "storm_count"},
                        {"label": "Severity Score", "value": "severity"},
                        {"label": "Fatalities", "value": "fatalities"},
                        {"label": "Total Damage (USD)", "value": "total_damage_usd"},
                    ],
                    value="storm_count",
                    clearable=False,
                ),
            ]),
            html.Div(className="control", children=[
                html.Label("Grid metric", className="small"),
                dcc.Dropdown(
                    id="stress_metric",
                    options=[{"label": k, "value": k} for k in STRESS_OPTIONS.keys()],
                    value="Typical Grid Forecast Miss",
                    clearable=False,
                ),
            ]),
            html.Div(className="control", children=[
                html.Label("Spike markers", className="small"),
                dcc.Dropdown(
                    id="spike_markers",
                    options=[
                        {"label": "Show Storm Count spikes", "value": "count"},
                        {"label": "Show Severity spikes", "value": "severity"},
                    ],
                    value=["count", "severity"],  # default: both
                    multi=True,
                    clearable=False,
                ),
            ]),
        ]),

        html.Div(id="drill_kpis", className="kpis"),
        html.Hr(className="sep"),

        dcc.Graph(
            id="timeline",
            style={"height": "560px"},
            config={"displayModeBar": False},
        ),

        # ---- NEW small multiples section ----
        html.Hr(className="sep"),
        html.Div(
            style={"marginTop": "10px"},
            children=[
                html.H4("Regional Patterns", style={"marginBottom": "4px"}),
                html.P(
                    "Monthly storm count (solid gold) vs grid anomaly (dashed cyan) across all regions.",
                    className="small",
                    style={"marginTop": "0px"},
                ),
                dcc.Graph(
                    id="regional_small_multiples",
                    style={"height": "720px"},
                    config={"displayModeBar": False},
                ),
            ],
        ),
    ])
])


# -------------------------
# Callback
# -------------------------
@dash.callback(
    Output("timeline", "figure"),
    Output("drill_kpis", "children"),
    Output("regional_small_multiples", "figure"),
    Input("custom_region", "value"),
    Input("storm_metric", "value"),
    Input("stress_metric", "value"),
    Input("spike_markers", "value"),
)
def update(region, storm_metric, stress_metric, spike_markers):
    fig = make_timeline(region, storm_metric, stress_metric, spike_markers)

    k = kpis_for_region(region)

    s_dates, s_vals = k["storms_series"]
    f_dates, f_vals = k["fatalities_series"]
    d_dates, d_vals = k["damage_series"]

    kpi_children = [
        kpi_card(
            "Total Storms",
            f"{k['storms_total']:,}",
            _sparkline_figure(s_dates, s_vals),
        ),
        kpi_card(
            "Total Fatalities",
            f"{k['fatalities_total']:,}",
            _sparkline_figure(f_dates, f_vals),
        ),
        kpi_card(
            "Total Damage (USD)",
            f"${k['damage_total']:,.0f}",
            _sparkline_figure(d_dates, d_vals),
        ),
    ]

    # small multiples reflect the selected grid metric (approved image behavior)
    sm_fig = make_regional_small_multiples(stress_metric)

    return fig, kpi_children, sm_fig