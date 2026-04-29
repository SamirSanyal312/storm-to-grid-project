import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.graph_objects as go

dash.register_page(__name__, path="/drilldown")

storms_region = pd.read_pickle("data/processed/storms_region_daily.pkl")
eia_daily = pd.read_pickle("data/processed/region_daily.pkl")

storms_region["date"] = pd.to_datetime(storms_region["date"])
eia_daily["date"] = pd.to_datetime(eia_daily["date"])

CUSTOM_REGIONS = sorted(storms_region["region"].dropna().unique().tolist())
EIA_REGIONS_AVAILABLE = sorted(eia_daily["Region"].dropna().unique().tolist())

# ---- Map your custom regions -> EIA regions (adjustable + defendable) ----
# NOTE: We only include entries that actually exist in the EIA file at runtime.
CUSTOM_TO_EIA_CANDIDATES = {
    "NE": ["NE"],
    "SE": ["SE"],
    "MIDW": ["MIDW"],
    "TX": ["TEX", "TX"],     # EIA sometimes uses TEX; keep both as candidates
    "PLAINS": ["CENT", "CENTRAL", "SWPP"],  # depends on your export
    "SW": ["SW"],
    "NW": ["NW"],
    "PAC": ["CAL", "CA", "NW"],  # PAC can be CAL; keep candidates
}

def eia_for_custom_region(custom_region: str) -> pd.DataFrame:
    candidates = CUSTOM_TO_EIA_CANDIDATES.get(custom_region, [])
    use = [r for r in candidates if r in EIA_REGIONS_AVAILABLE]

    # If nothing matches, fall back to national total (still useful context)
    if not use:
        g = eia_daily.groupby("date", as_index=False)["Demand Anomaly (MW)"].sum()
        g["Region"] = "NATIONAL"
        return g

    return eia_daily[eia_daily["Region"].isin(use)].groupby("date", as_index=False).agg({
        "Demand Anomaly (MW)": "sum"
    })

def make_timeline(custom_region: str, storm_metric: str):
    s = storms_region[storms_region["region"] == custom_region].sort_values("date")
    e = eia_for_custom_region(custom_region).sort_values("date")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=s["date"], y=s[storm_metric],
        name=f"Storm {storm_metric}",
        mode="lines",
        yaxis="y1"
    ))

    fig.add_trace(go.Scatter(
        x=e["date"], y=e["Demand Anomaly (MW)"],
        name="Demand Anomaly (MW)",
        mode="lines",
        yaxis="y2"
    ))

    fig.update_layout(
        title=f"Storm → Grid (2024) — Region: {custom_region}",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), showgrid=False),
        yaxis=dict(title=f"{storm_metric}", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(title="Demand Anomaly (MW)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    return fig

def kpis_for_region(custom_region: str):
    s = storms_region[storms_region["region"] == custom_region]
    return {
        "storms": int(s["storm_count"].sum()),
        "fatalities": int(s["fatalities"].sum()),
        "damage": float(s["total_damage_usd"].sum()),
        "severity": float(s["severity"].sum()),
    }

layout = html.Div([
    html.Div(className="card", children=[
        html.H3("Chapter 2 — Drilldown: Storm → Grid timeline"),
        html.P(
            "Filter by your custom region and compare storm intensity against grid stress (demand anomaly). "
            "This uses linked context and a time-range slider to support analysis.",
            className="small"
        ),

        html.Div(className="controls", children=[
            html.Div(className="control", children=[
                html.Label("Custom region", className="small"),
                dcc.Dropdown(
                    id="custom_region",
                    options=[{"label": r, "value": r} for r in CUSTOM_REGIONS],
                    value=CUSTOM_REGIONS[0] if CUSTOM_REGIONS else None,
                    clearable=False
                ),
            ]),
            html.Div(className="control", children=[
                html.Label("Storm metric", className="small"),
                dcc.Dropdown(
                    id="storm_metric",
                    options=[
                        {"label": "Storm Count", "value": "storm_count"},
                        {"label": "Severity", "value": "severity"},
                        {"label": "Fatalities", "value": "fatalities"},
                        {"label": "Total Damage (USD)", "value": "total_damage_usd"},
                    ],
                    value="storm_count",
                    clearable=False
                ),
            ]),
        ]),

        html.Div(id="drill_kpis", className="kpis"),
        html.Hr(className="sep"),

        dcc.Graph(id="timeline", style={"height": "560px"}, config={"displayModeBar": False}),
        html.P(
            f"EIA regions available in your file: {', '.join(EIA_REGIONS_AVAILABLE[:12])}"
            + ("..." if len(EIA_REGIONS_AVAILABLE) > 12 else ""),
            className="small"
        )
    ])
])

@dash.callback(
    Output("timeline", "figure"),
    Output("drill_kpis", "children"),
    Input("custom_region", "value"),
    Input("storm_metric", "value"),
)
def update(custom_region, storm_metric):
    fig = make_timeline(custom_region, storm_metric)

    k = kpis_for_region(custom_region)
    kpi_children = [
        html.Div(className="kpi", children=[html.Div("Total Storms", className="label"), html.Div(f"{k['storms']:,}", className="value")]),
        html.Div(className="kpi", children=[html.Div("Total Fatalities", className="label"), html.Div(f"{k['fatalities']:,}", className="value")]),
        html.Div(className="kpi", children=[html.Div("Total Damage (USD)", className="label"), html.Div(f"${k['damage']:,.0f}", className="value")]),
    ]
    return fig, kpi_children