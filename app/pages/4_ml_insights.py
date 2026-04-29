# app/pages/4_ml_insights.py
import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dash.register_page(__name__, path="/ml", name="ML Insights")

# -------------------------
# Paths to processed data
# -------------------------
STORMS_REG_DAILY = "data/processed/storms_region_daily.pkl"
EIA_REG_DAILY = "data/processed/eia_custom_region_daily.pkl"

# -------------------------
# Load + normalize daily data
# -------------------------
def load_daily_merged():
    storms = pd.read_pickle(STORMS_REG_DAILY).copy()
    eia = pd.read_pickle(EIA_REG_DAILY).copy()

    storms.columns = [c.strip().lower().replace(" ", "_") for c in storms.columns]
    eia.columns = [c.strip().lower().replace(" ", "_") for c in eia.columns]

    # storms: region -> custom_region
    if "custom_region" not in storms.columns:
        if "region" in storms.columns:
            storms = storms.rename(columns={"region": "custom_region"})
        else:
            raise ValueError(f"storms_region_daily.pkl missing region/custom_region. cols={storms.columns.tolist()}")

    # eia: ensure custom_region
    if "custom_region" not in eia.columns:
        if "region" in eia.columns:
            eia = eia.rename(columns={"region": "custom_region"})
        else:
            raise ValueError(f"eia_custom_region_daily.pkl missing region/custom_region. cols={eia.columns.tolist()}")

    if "date" not in storms.columns:
        raise ValueError(f"storms_region_daily.pkl missing date. cols={storms.columns.tolist()}")
    if "date" not in eia.columns:
        raise ValueError(f"eia_custom_region_daily.pkl missing date. cols={eia.columns.tolist()}")

    storms["date"] = pd.to_datetime(storms["date"], errors="coerce")
    eia["date"] = pd.to_datetime(eia["date"], errors="coerce")
    storms = storms.dropna(subset=["date"])
    eia = eia.dropna(subset=["date"])

    storms["custom_region"] = storms["custom_region"].astype(str).str.strip()
    eia["custom_region"] = eia["custom_region"].astype(str).str.strip()

    # normalize EIA anomaly column names
    rename_map = {
        "anomaly_abs_mean_mw": "mean_abs_anomaly_mw",
        "anomaly_abs_max_mw": "max_abs_anomaly_mw",
        "anomaly_mean_mw": "signed_mean_anomaly_mw",
    }
    eia = eia.rename(columns=rename_map)

    storm_cols = [c for c in ["storm_count", "severity", "fatalities", "total_damage_usd"] if c in storms.columns]
    if not storm_cols:
        raise ValueError(f"No expected storm columns found. cols={storms.columns.tolist()}")

    eia_cols = [c for c in ["mean_abs_anomaly_mw", "max_abs_anomaly_mw", "signed_mean_anomaly_mw"] if c in eia.columns]
    if not eia_cols:
        raise ValueError(f"No expected EIA columns found after renaming. cols={eia.columns.tolist()}")

    storms = storms[["custom_region", "date"] + storm_cols]
    eia = eia[["custom_region", "date"] + eia_cols]

    df = storms.merge(eia, on=["custom_region", "date"], how="inner").sort_values(["custom_region", "date"])
    return df


DF = load_daily_merged()
REGIONS = sorted(DF["custom_region"].dropna().unique().tolist())

GRID_TARGETS = {
    "Grid stress: mean |anomaly| (MW)": "mean_abs_anomaly_mw",
    "Grid stress: max |anomaly| (MW)": "max_abs_anomaly_mw",
    "Grid bias: signed mean anomaly (MW)": "signed_mean_anomaly_mw",
}
GRID_LABELS = {v: k for k, v in GRID_TARGETS.items()}

STORM_FEATURE_SETS = {
    "Storm Count only": ["storm_count"],
    "Count + Severity": ["storm_count", "severity"],
    "All storm features": ["storm_count", "severity", "fatalities", "total_damage_usd"],
}

# -------------------------
# ML helpers
# -------------------------
def make_lag_features(df_region, base_cols, lags=7):
    out = df_region[["date"]].copy()
    for c in base_cols:
        if c not in df_region.columns:
            continue
        for k in range(lags + 1):
            out[f"{c}_lag{k}"] = df_region[c].shift(k)
    return out

def fit_ridge_lags(df_region, y_col, base_cols, lags=7):
    """
    Ridge:
      y(t) ~ storms(t) + storms(t-1) + ... storms(t-lags)
    We standardize BOTH X and y so coefficients are comparable + readable.
    """
    df_region = df_region.sort_values("date").copy()
    lagged = make_lag_features(df_region, base_cols, lags=lags)

    X = lagged.drop(columns=["date"])
    y = df_region[y_col].values.astype(float)

    valid = ~X.isna().any(axis=1) & np.isfinite(y)
    X = X.loc[valid]
    y = y[valid]

    if len(X) < 30:
        return pd.DataFrame(columns=["feature", "coef"]), None

    # ✅ Standardize y to make coefficients interpretable/scale-free
    y = (y - np.mean(y)) / (np.std(y) + 1e-9)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    model.fit(X, y)
    coefs = model.named_steps["ridge"].coef_
    coef_df = pd.DataFrame({"feature": X.columns, "coef": coefs})

    base0 = base_cols[0]
    subset = coef_df[coef_df["feature"].str.startswith(f"{base0}_lag")]
    if len(subset) == 0:
        best_lag = None
    else:
        # choose best positive association if possible, else max abs
        pos = subset[subset["coef"] > 0]
        pick = pos if len(pos) else subset
        best_feat = pick.iloc[pick["coef"].abs().argmax()]["feature"]
        best_lag = int(best_feat.split("lag")[-1])

    return coef_df.sort_values("coef", ascending=False), best_lag

def fit_isolation_anomalies(df_region, y_col, storm_cols, contamination=0.03):
    cols = [c for c in storm_cols if c in df_region.columns] + [y_col]
    X = df_region[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < 40:
        return pd.DataFrame()

    iso = IsolationForest(
        n_estimators=350,
        contamination=contamination,
        random_state=42
    )
    iso.fit(X.values)
    scores = -iso.score_samples(X.values)  # higher = more anomalous

    out = df_region.loc[X.index, ["date", "custom_region"] + cols].copy()
    out["anomaly_score"] = scores
    out = out.sort_values("anomaly_score", ascending=False)
    return out

def region_lag_heatmap(df, y_col, base_cols, lags=7):
    """
    Build region x lag matrix of ridge coefficients for the FIRST storm feature in base_cols.
    """
    base0 = base_cols[0]
    rows = []
    for r in sorted(df["custom_region"].unique()):
        df_r = df[df["custom_region"] == r].copy().sort_values("date")
        coef_df, _ = fit_ridge_lags(df_r, y_col=y_col, base_cols=base_cols, lags=lags)
        if len(coef_df) == 0:
            continue
        sub = coef_df[coef_df["feature"].str.startswith(f"{base0}_lag")].copy()
        if len(sub) == 0:
            continue
        sub["lag"] = sub["feature"].str.extract(r"lag(\d+)").astype(int)
        vals = {int(row["lag"]): float(row["coef"]) for _, row in sub.iterrows()}
        row_out = {"custom_region": r}
        for k in range(lags + 1):
            row_out[k] = vals.get(k, np.nan)
        rows.append(row_out)

    mat = pd.DataFrame(rows)
    if len(mat) == 0:
        return px.imshow([[0]], title="Heatmap unavailable (insufficient data).")

    mat = mat.set_index("custom_region").sort_index()
    z = mat.values
    # symmetric color scale around 0 helps interpret direction
    vmax = float(np.nanmax(np.abs(z))) if np.isfinite(np.nanmax(np.abs(z))) else 1.0
    vmax = max(vmax, 0.25)

    fig = px.imshow(
        mat,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-vmax,
        zmax=vmax,
        labels=dict(x="Lag days (storm at t-lag)", y="Custom region", color="Coef"),
        title=f"Region × Lag sensitivity heatmap — target: {GRID_LABELS.get(y_col, y_col)}",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig

def _plot_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=10, r=10, t=60, b=45),
    )
    return fig

# -------------------------
# Layout
# -------------------------
layout = html.Div([
    html.Div(className="card", children=[
        html.Div(className="section-head", children=[
            html.H3("ML Insights — Storm → Grid (Exploratory, not causal)"),
            html.P(
                "ML adds: (1) anomaly detection to surface unusual storm+grid days, and "
                "(2) lag sensitivity (Ridge) to quantify whether grid stress aligns after storm activity.",
                className="small"
            ),
        ]),

        html.Div(className="controls", children=[
            html.Div(className="control", children=[
                html.Label("Custom region", className="small"),
                dcc.Dropdown(
                    id="ml_region",
                    options=[{"label": r, "value": r} for r in REGIONS],
                    value=REGIONS[0] if REGIONS else None,
                    clearable=False,
                )
            ]),
            html.Div(className="control", children=[
                html.Label("Grid target (EIA)", className="small"),
                dcc.Dropdown(
                    id="ml_target",
                    options=[{"label": k, "value": v} for k, v in GRID_TARGETS.items()],
                    value="mean_abs_anomaly_mw",
                    clearable=False,
                )
            ]),
            html.Div(className="control", children=[
                html.Label("Storm features", className="small"),
                dcc.Dropdown(
                    id="ml_features",
                    options=[{"label": k, "value": k} for k in STORM_FEATURE_SETS.keys()],
                    value="Count + Severity",
                    clearable=False,
                )
            ]),
        ]),

        html.Div(className="kpis", children=[
            html.Div(className="kpi", children=[
                html.Div("Best lag (days)", className="label"),
                html.Div(id="best_lag_kpi", className="value"),
            ]),
            html.Div(className="kpi", children=[
                html.Div("Top anomaly day", className="label"),
                html.Div(id="top_anom_kpi", className="value"),
            ]),
            html.Div(className="kpi", children=[
                html.Div("Interpretation", className="label"),
                html.Div(id="interpret_kpi", className="value kpi-long"),
            ]),
        ]),

        html.Hr(className="sep"),

        html.H4("1) Lag sensitivity (Ridge)"),
        html.P("Coefficients show association strength for storm(t−lag) predicting the selected grid target. (Not causality.)", className="small"),
        dcc.Graph(id="lag_coef_plot", config={"displayModeBar": False}),

        html.Hr(className="sep"),

        html.H4("2) ML-discovered anomaly days (Isolation Forest)"),
        html.P("Anomalies represent unusual combinations of storm activity + grid behavior.", className="small"),
        dcc.Graph(id="anom_scatter", config={"displayModeBar": False}),
        html.Div(id="anom_table", className="small", style={"marginTop": "12px"}),

        html.Hr(className="sep"),

        html.H4("3) Region × Lag heatmap (presentation visual)"),
        html.P("Compare lag response patterns across all regions for the first selected storm feature.", className="small"),
        dcc.Graph(id="lag_heatmap", config={"displayModeBar": False}),
    ])
])

# -------------------------
# Callback
# -------------------------
@dash.callback(
    Output("lag_coef_plot", "figure"),
    Output("anom_scatter", "figure"),
    Output("anom_table", "children"),
    Output("best_lag_kpi", "children"),
    Output("top_anom_kpi", "children"),
    Output("interpret_kpi", "children"),
    Output("lag_heatmap", "figure"),
    Input("ml_region", "value"),
    Input("ml_target", "value"),
    Input("ml_features", "value"),
)
def update_ml(region, y_col, feature_set_name):
    target_label = GRID_LABELS.get(y_col, y_col)

    if region is None:
        empty = _plot_theme(px.line(title="No region available"))
        return empty, empty, "", "—", "—", "—", empty

    storm_cols = [c for c in STORM_FEATURE_SETS[feature_set_name] if c in DF.columns]
    if not storm_cols:
        empty = _plot_theme(px.line(title="No storm features available in data"))
        return empty, empty, "", "—", "—", "—", empty

    df_r = DF[DF["custom_region"] == region].copy().sort_values("date")

    # --- Ridge lag model
    coef_df, best_lag = fit_ridge_lags(df_r, y_col=y_col, base_cols=storm_cols, lags=7)

    if len(coef_df) == 0:
        lag_fig = _plot_theme(px.line(title="Not enough valid days for lag model (need ~30+)."))
        best_lag_txt = "—"
    else:
        base0 = storm_cols[0]
        plot_df = coef_df[coef_df["feature"].str.startswith(f"{base0}_lag")].copy()
        plot_df["lag"] = plot_df["feature"].str.extract(r"lag(\d+)").astype(int)

        lag_fig = px.bar(
            plot_df.sort_values("lag"),
            x="lag",
            y="coef",
            title=f"Lag sensitivity for {region} — target: {target_label}",
            labels={"lag": "Lag days (storm at t-lag)", "coef": "Standardized coefficient"},
        )
        lag_fig = _plot_theme(lag_fig)
        lag_fig.update_xaxes(dtick=1)

        best_lag_txt = str(best_lag) if best_lag is not None else "—"

    # --- Isolation Forest anomalies
    anom = fit_isolation_anomalies(df_r, y_col=y_col, storm_cols=storm_cols, contamination=0.03)

    if len(anom) == 0:
        anom_fig = _plot_theme(px.scatter(title="Not enough data for anomaly detection."))
        top_anom_txt = "—"
        table_html = ""
    else:
        top = anom.iloc[0]
        top_anom_txt = f"{top['date'].date()} (score={top['anomaly_score']:.2f})"

        # Better anomaly plot: storms vs target, size = anomaly score
        x_feature = "storm_count" if "storm_count" in anom.columns else storm_cols[0]
        anom_fig = px.scatter(
            anom.head(220),
            x=x_feature,
            y=y_col,
            size="anomaly_score",
            hover_data=["date", "custom_region"] + storm_cols + [y_col, "anomaly_score"],
            title=f"Anomaly space in {region}: {x_feature} vs {target_label} (size = anomaly score)",
            labels={x_feature: x_feature.replace("_", " ").title(), y_col: target_label, "anomaly_score": "Anomaly score"},
        )
        anom_fig = _plot_theme(anom_fig)

        # Table (top 10)
        tbl = anom.head(10)[["date", "anomaly_score"] + storm_cols + [y_col]].copy()
        tbl["date"] = tbl["date"].dt.date.astype(str)
        tbl["anomaly_score"] = tbl["anomaly_score"].map(lambda v: f"{v:.2f}")
        for c in storm_cols + [y_col]:
            tbl[c] = tbl[c].map(lambda v: "—" if pd.isna(v) else f"{float(v):,.0f}")

        table_html = html.Div(className="table-wrap", children=[
            html.Div("Top 10 anomaly days", className="small", style={"marginBottom": "8px"}),
            html.Table(
                className="data-table",
                children=[
                    html.Thead(html.Tr([html.Th(c) for c in tbl.columns])),
                    html.Tbody([
                        html.Tr([html.Td(tbl.iloc[i][c]) for c in tbl.columns]) for i in range(len(tbl))
                    ]),
                ],
            )
        ])

    # Interpretation KPI
    if best_lag is None:
        interpret = "Lag model: insufficient data."
    elif best_lag == 0:
        interpret = "Strongest association is same-day (exploratory; not causal)."
    else:
        interpret = f"Grid stress tends to follow storms by ~{best_lag} day(s) (association, not causality)."

    # --- Heatmap across regions (based on FIRST storm feature)
    heat_fig = region_lag_heatmap(DF, y_col=y_col, base_cols=storm_cols, lags=7)

    return lag_fig, anom_fig, table_html, best_lag_txt, top_anom_txt, interpret, heat_fig