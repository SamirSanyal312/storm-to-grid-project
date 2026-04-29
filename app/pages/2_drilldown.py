import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.graph_objects as go

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

    # Optional: spike markers (both types supported)
    spike_markers = spike_markers or []
    if spike_markers:
        add_spike_markers(fig, region, spike_markers, top_n=5)

    return fig

def kpis_for_region(region: str):
    s = storms_region[storms_region["region"] == region]
    return {
        "storms": int(s["storm_count"].sum()),
        "fatalities": int(s["fatalities"].sum()),
        "damage": float(s["total_damage_usd"].sum()),
    }

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
    ])
])

@dash.callback(
    Output("timeline", "figure"),
    Output("drill_kpis", "children"),
    Input("custom_region", "value"),
    Input("storm_metric", "value"),
    Input("stress_metric", "value"),
    Input("spike_markers", "value"),
)
def update(region, storm_metric, stress_metric, spike_markers):
    fig = make_timeline(region, storm_metric, stress_metric, spike_markers)
    k = kpis_for_region(region)

    kpi_children = [
        html.Div(className="kpi", children=[
            html.Div("Total Storms", className="label"),
            html.Div(f"{k['storms']:,}", className="value"),
        ]),
        html.Div(className="kpi", children=[
            html.Div("Total Fatalities", className="label"),
            html.Div(f"{k['fatalities']:,}", className="value"),
        ]),
        html.Div(className="kpi", children=[
            html.Div("Total Damage (USD)", className="label"),
            html.Div(f"${k['damage']:,.0f}", className="value"),
        ]),
    ]
    return fig, kpi_children
