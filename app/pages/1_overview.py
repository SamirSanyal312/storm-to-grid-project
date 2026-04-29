import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px

dash.register_page(__name__, path="/")

# ---------------- Data ----------------
state_month = pd.read_pickle("data/processed/state_month.pkl")
state_month["month"] = state_month["month"].astype(str)

outliers = pd.read_pickle("data/processed/outliers_state_day.pkl")
outliers["date"] = pd.to_datetime(outliers["date"], errors="coerce")
outliers["month"] = outliers["date"].dt.to_period("M").astype(str)

METRICS = {
    "Storm Count": "storm_count",
    "Severity Score": "severity",
    "Fatalities": "fatalities",
    "Total Damage (USD)": "total_damage_usd",
}

def make_map(metric_col: str, start_month: str | None = None):
    d = state_month.copy()

    # Colorblind-friendly sequential palette
    colorscale = "cividis"

    hover = {
        "month": True,
        "storm_count": True,
        "fatalities": True,
        "severity": ":,.2f",
        "total_damage_usd": ":,.0f",
        "state_abbr": False,
    }

    title_metric = metric_col
    color_col = metric_col
    colorbar_tickformat = ",.0f"

    # Log-transform for heavy-tailed damage
    if metric_col == "total_damage_usd":
        d["map_value"] = np.log10(d["total_damage_usd"].fillna(0) + 1)
        color_col = "map_value"
        title_metric = "log10(Total Damage USD + 1)"
        colorbar_tickformat = ".2f"

    fig = px.choropleth(
        d,
        locations="state_abbr",
        locationmode="USA-states",
        color=color_col,
        animation_frame="month",
        scope="usa",
        color_continuous_scale=colorscale,
        hover_data=hover,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#e5e7eb"),
        title=f"Animated Choropleth (2024) — {title_metric}",
        coloraxis_colorbar=dict(title="Value", tickformat=colorbar_tickformat),
    )
    fig.update_traces(marker_line_width=0.35)

    # ---- Jump animation to a chosen month (story click) ----
    if start_month:
        # Plotly stores animation frames by month string
        # We set the "active" slider index to that month if found
        months = sorted(d["month"].unique().tolist())
        if start_month in months and fig.layout.sliders:
            idx = months.index(start_month)
            fig.layout.sliders[0].active = idx

    return fig

def outlier_table(n=8):
    # Pick top N rows and show as clickable list
    o = outliers.sort_values("severity", ascending=False).head(n).copy()
    rows = []
    for i, r in o.iterrows():
        label = f"{r['date'].date()} — {r['state_abbr']} | severity={r['severity']:.1f} | storms={int(r['storm_count'])}"
        rows.append(
            html.Li(
                html.Button(
                    label,
                    id={"type": "outlier-btn", "index": int(i)},
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "textAlign": "left",
                        "background": "rgba(0,0,0,0.18)",
                        "border": "1px solid rgba(255,255,255,0.10)",
                        "color": "#e5e7eb",
                        "padding": "10px",
                        "borderRadius": "12px",
                        "cursor": "pointer",
                        "marginBottom": "8px",
                    },
                )
            )
        )
    return html.Ul(rows, style={"listStyleType": "none", "paddingLeft": 0, "margin": 0})

layout = html.Div([
    dcc.Store(id="selected-month", data=None),

    html.Div(className="card", children=[
        html.H3("Chapter 1 — Animated choropleth (2024)"),
        html.P(
            "Overview-first: watch storm intensity (and impact) move across the U.S. month-by-month. "
            "Palette: Cividis (colorblind-friendly). Damage uses log scale for readability.",
            className="small"
        ),

        html.Div(className="controls", children=[
            html.Div(className="control", children=[
                html.Label("Metric", className="small"),
                dcc.Dropdown(
                    id="metric",
                    options=[{"label": k, "value": v} for k, v in METRICS.items()],
                    value="storm_count",
                    clearable=False
                )
            ])
        ]),

        dcc.Graph(
            id="choropleth",
            figure=make_map("storm_count", None),
            style={"height": "560px"},
            config={"displayModeBar": False}
        ),
    ]),

    html.Div(className="card", children=[
        html.H3("Story beats — top extreme state-days (click to jump)"),
        html.P(
            "Details-on-demand: these are the most severe state-days in 2024. "
            "Click one to jump the map animation to that month.",
            className="small"
        ),
        html.Div(id="outlier-list", children=outlier_table(8)),
        #html.Div(id="outlier-selected", className="small", style={"marginTop": "10px"})
        dcc.Markdown(id="outlier-selected", className="small", style={"marginTop": "10px"})
    ])
])

# ---------- Callbacks ----------
@dash.callback(
    Output("selected-month", "data"),
    Output("outlier-selected", "children"),
    Input({"type": "outlier-btn", "index": dash.ALL}, "n_clicks"),
    State({"type": "outlier-btn", "index": dash.ALL}, "id"),
    prevent_initial_call=True
)
def pick_outlier(n_clicks, ids):
    if not n_clicks or max(n_clicks) == 0:
        return dash.no_update, dash.no_update

    clicked_pos = int(np.argmax(n_clicks))
    idx = ids[clicked_pos]["index"]

    row = outliers.loc[idx]
    month = row["month"]
    #msg = (
        #f"Selected: {row['date'].date()} in {row['state_abbr']} "
        #f"(storms={int(row['storm_count'])}, fatalities={int(row['fatalities'])}, "
        #f"damage=${row['total_damage_usd']:,.0f})"
    #)
    msg = (
    f"**Case study selected** → {row['date'].date()} ({month}) in **{row['state_abbr']}** | "
    f"storms={int(row['storm_count'])} | fatalities={int(row['fatalities'])} | "
    f"damage=${row['total_damage_usd']:,.0f} | severity={row['severity']:.1f}"
)
    return month, msg

@dash.callback(
    Output("choropleth", "figure"),
    Input("metric", "value"),
    Input("selected-month", "data"),
)
def update_map(metric_col, selected_month):
    return make_map(metric_col, selected_month)