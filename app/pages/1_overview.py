# app/pages/1_overview.py
import base64
from io import BytesIO

import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

dash.register_page(__name__, path="/")

# -------------------------
# Paths
# -------------------------
STATE_TO_REGION_PATH = "data/processed/state_to_region.csv"
EIA_DAILY_PATH = "data/processed/eia_custom_region_daily.pkl"

# -------------------------
# Load processed data
# -------------------------
state_month_raw = pd.read_pickle("data/processed/state_month.pkl").copy()
state_month_raw["month"] = state_month_raw["month"].astype(str)
state_month_raw["state_abbr"] = state_month_raw["state_abbr"].astype(str).str.strip().str.upper()

outliers = pd.read_pickle("data/processed/outliers_state_day.pkl").copy()
outliers["date"] = pd.to_datetime(outliers["date"], errors="coerce")
outliers["month"] = outliers["date"].dt.to_period("M").astype(str)

event_type_month = pd.read_pickle("data/processed/event_type_month.pkl").copy()
event_type_month["month"] = event_type_month["month"].astype(str)

# -------------------------
# UI labels / metric keys
# -------------------------
METRICS = {
    "Storm Count": "storm_count",
    "Severity Score": "severity",
    "Fatalities": "fatalities",
    "Total Damage (USD)": "total_damage_usd",
}

# ✅ IMPORTANT: Dash dropdown can't use None -> use "none" sentinel
GRID_METRICS = {
    "None (NOAA only)": "none",
    "Grid stress: mean |anomaly| (MW)": "mean_abs_anomaly_mw",
    "Grid stress: max |anomaly| (MW)": "max_abs_anomaly_mw",
    "Grid bias: signed mean anomaly (MW)": "signed_mean_anomaly_mw",
}

VIS_LABELS = {
    "month": "Month",
    "state_abbr": "State",
    "storm_count": "Storm Count",
    "severity": "Severity Score",
    "fatalities": "Fatalities",
    "total_damage_usd": "Total Damage (USD)",
    "map_value": "Damage Intensity (log scale)",
    "custom_region": "EIA Region",
    "mean_abs_anomaly_mw": "Grid stress: mean |anomaly| (MW)",
    "max_abs_anomaly_mw": "Grid stress: max |anomaly| (MW)",
    "signed_mean_anomaly_mw": "Grid bias: signed mean anomaly (MW)",
}

def display_label(col: str) -> str:
    return VIS_LABELS.get(col, col.replace("_", " ").title())

WC_MONTHS = ["ALL"] + sorted(event_type_month["month"].unique().tolist())

# -------------------------
# State centroids (for prism spike anchors)
# -------------------------
STATE_CENTROIDS = {
    "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419), "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123), "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371), "DE": (39.318523, -75.507141), "DC": (38.897438, -77.026817),
    "FL": (27.766279, -81.686783), "GA": (33.040619, -83.643074), "HI": (21.094318, -157.498337),
    "ID": (44.240459, -114.478828), "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526), "KS": (38.526600, -96.726486), "KY": (37.668140, -84.670067),
    "LA": (31.169546, -91.867805), "ME": (44.693947, -69.381927), "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106), "MI": (43.326618, -84.536095), "MN": (45.694454, -93.900192),
    "MS": (32.741646, -89.678696), "MO": (38.456085, -92.288368), "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082), "NV": (38.313515, -117.055374), "NH": (43.452492, -71.563896),
    "NJ": (40.298904, -74.521011), "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419), "ND": (47.528912, -99.784012), "OH": (40.388783, -82.764915),
    "OK": (35.565342, -96.928917), "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780), "SC": (33.856892, -80.945007), "SD": (44.299782, -99.438828),
    "TN": (35.747845, -86.692345), "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686), "VA": (37.769337, -78.169968), "WA": (47.400902, -121.490494),
    "WV": (38.491226, -80.954453), "WI": (44.268543, -89.616508), "WY": (42.755966, -107.302490),
}

def _colors():
    # colorblind-friendly sequential
    return "cividis"

def _set_slider_to_month(fig, month_value: str | None, months: list[str]):
    if not month_value or not fig.layout.sliders:
        return
    if month_value in months:
        fig.layout.sliders[0].active = months.index(month_value)

# -------------------------
# Robust state_to_region.csv loader
# -------------------------
def load_state_to_region():
    m = pd.read_csv(STATE_TO_REGION_PATH)
    m.columns = [c.strip() for c in m.columns]

    # find state column
    state_candidates = ["state_abbr", "state", "abbr", "STATE", "State", "STATE_ABBR"]
    state_col = next((c for c in state_candidates if c in m.columns), None)
    if state_col is None:
        raise ValueError(
            f"state_to_region.csv must contain a state column like 'state_abbr'. "
            f"Found columns: {m.columns.tolist()}"
        )

    # find region column
    region_candidates = [
        "custom_region", "region", "Region", "EIA Region", "EIA_Region",
        "eia_region", "customRegion", "CUSTOM_REGION"
    ]
    region_col = next((c for c in region_candidates if c in m.columns), None)

    # if not found, assume 2-column file and pick the other col
    if region_col is None:
        other_cols = [c for c in m.columns if c != state_col]
        if len(other_cols) == 1:
            region_col = other_cols[0]
        else:
            raise ValueError(
                f"Couldn't infer region column in state_to_region.csv. "
                f"Found columns: {m.columns.tolist()}"
            )

    m = m.rename(columns={state_col: "state_abbr", region_col: "custom_region"})
    m["state_abbr"] = m["state_abbr"].astype(str).str.strip().str.upper()
    m["custom_region"] = m["custom_region"].astype(str).str.strip()
    return m[["state_abbr", "custom_region"]]

# -------------------------
# EIA loader (normalize columns + rename to standard names)
# -------------------------
def load_region_month_grid():
    df = pd.read_pickle(EIA_DAILY_PATH).copy()

    # normalize colnames: lower, strip, spaces->underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # normalize date
    if "date" not in df.columns:
        raise ValueError(f"EIA file missing 'date'. Columns seen: {sorted(df.columns.tolist())}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # normalize region col
    if "custom_region" not in df.columns and "region" in df.columns:
        df = df.rename(columns={"region": "custom_region"})
    if "custom_region" not in df.columns:
        raise ValueError(f"EIA file missing 'custom_region'. Columns seen: {sorted(df.columns.tolist())}")
    df["custom_region"] = df["custom_region"].astype(str).str.strip()

    # month
    if "month" not in df.columns:
        #df["month"] = df["date"].dt.to_period("m").astype(str)
        df["month"] = df["date"].dt.to_period("M").astype(str)
    else:
        df["month"] = df["month"].astype(str)

    # ✅ Your file uses: anomaly_abs_mean_mw, anomaly_abs_max_mw, anomaly_mean_mw
    rename_map = {
        "anomaly_abs_mean_mw": "mean_abs_anomaly_mw",
        "anomaly_abs_max_mw": "max_abs_anomaly_mw",
        "anomaly_mean_mw": "signed_mean_anomaly_mw",
    }
    df = df.rename(columns=rename_map)

    agg_cols = [c for c in ["mean_abs_anomaly_mw", "max_abs_anomaly_mw", "signed_mean_anomaly_mw"] if c in df.columns]
    if not agg_cols:
        raise ValueError(
            "Could not find expected EIA anomaly columns after renaming. "
            f"Columns seen: {sorted(df.columns.tolist())}"
        )

    g = df.groupby(["custom_region", "month"], as_index=False)[agg_cols].mean()
    return g

def build_state_month_enriched():
    base = state_month_raw.copy()
    s2r = load_state_to_region()
    rmonth = load_region_month_grid()
    base = base.merge(s2r, on="state_abbr", how="left")
    base = base.merge(rmonth, on=["custom_region", "month"], how="left")
    return base

STATE_MONTH = build_state_month_enriched()

# -------------------------
# Choropleth (NOAA color + EIA hover)
# -------------------------
def make_choropleth(metric_col: str, grid_metric_col: str | None, start_month: str | None = None):
    d = STATE_MONTH.copy()
    months = sorted(d["month"].unique().tolist())

    title_metric = display_label(metric_col)
    color_col = metric_col
    colorbar_title = title_metric
    tickfmt = ",.0f"

    if metric_col == "total_damage_usd":
        d["map_value"] = np.log10(d["total_damage_usd"].fillna(0) + 1)
        color_col = "map_value"
        title_metric = "Total Damage (USD)"
        colorbar_title = "Damage Intensity (log scale)"
        tickfmt = ".2f"

    hover_data = {
        "month": True,
        "storm_count": True,
        "fatalities": True,
        "severity": ":,.2f",
        "total_damage_usd": ":,.0f",
        "custom_region": True,
        "state_abbr": False,
    }
    if grid_metric_col is not None and grid_metric_col in d.columns:
        hover_data[grid_metric_col] = ":,.0f"

    fig = px.choropleth(
        d,
        locations="state_abbr",
        locationmode="USA-states",
        color=color_col,
        animation_frame="month",
        scope="usa",
        color_continuous_scale=_colors(),
        hover_data=hover_data,
        labels=VIS_LABELS,
    )

    _set_slider_to_month(fig, start_month, months)

    fig.update_layout(
        title=dict(
            text=f"Animated Choropleth — {title_metric}",
            x=0.02, xanchor="left", y=0.99, yanchor="top",
            font=dict(size=18, color="#e5e7eb"),
        ),
        margin=dict(l=0, r=0, t=105, b=125),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#e5e7eb"),
        coloraxis_colorbar=dict(title=colorbar_title, tickformat=tickfmt),
    )

    # controls below map
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].x = 0.02
        fig.layout.updatemenus[0].y = -0.10
        fig.layout.updatemenus[0].xanchor = "left"
        fig.layout.updatemenus[0].yanchor = "bottom"

    if fig.layout.sliders:
        fig.layout.sliders[0].x = 0.10
        fig.layout.sliders[0].y = -0.10
        fig.layout.sliders[0].xanchor = "left"
        fig.layout.sliders[0].yanchor = "bottom"
        fig.layout.sliders[0].len = 0.86
        fig.layout.sliders[0].currentvalue.prefix = "Month: "

    fig.update_traces(marker_line_width=0.35)
    return fig

# -------------------------
# Prism spikes (NOAA elevation + EIA outline strength)
# -------------------------
PRISM_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]
PRISM_COLORS = {
    "Very Low": "#1d3557",
    "Low": "#457b9d",
    "Medium": "#8d99ae",
    "High": "#c2b36f",
    "Very High": "#f4d35e",
}

def _assign_discrete_bins(series, n_bins=5):
    s = pd.Series(series).fillna(0).astype(float)
    if len(s) == 0:
        return pd.Series([], dtype=str), []
    if s.nunique() <= 1:
        return pd.Series(["Medium"] * len(s), index=s.index), [s.min(), s.max()]

    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(s.quantile(q).values)
    if len(edges) < 3 or len(edges) != n_bins + 1:
        edges = np.linspace(s.min(), s.max() + 1e-9, n_bins + 1)

    labels = PRISM_ORDER[: len(edges) - 1]
    cats = pd.cut(s, bins=edges, labels=labels, include_lowest=True, duplicates="drop").astype(str)
    cats = cats.replace("nan", labels[0] if labels else "Medium")
    return cats, edges

def _metric_preparation(df, metric_col):
    d = df.copy()
    if metric_col == "storm_count":
        d["metric_raw"] = d["storm_count"].fillna(0).astype(float)
        d["color_basis"] = d["metric_raw"]
        metric_title = "Storm Count"
        metric_fmt = ":,.0f"
    elif metric_col == "severity":
        d["metric_raw"] = d["severity"].fillna(0).astype(float)
        d["color_basis"] = d["metric_raw"]
        metric_title = "Severity Score"
        metric_fmt = ":,.1f"
    elif metric_col == "fatalities":
        d["metric_raw"] = d["fatalities"].fillna(0).astype(float)
        d["color_basis"] = d["metric_raw"]
        metric_title = "Fatalities"
        metric_fmt = ":,.0f"
    elif metric_col == "total_damage_usd":
        d["metric_raw"] = d["total_damage_usd"].fillna(0).astype(float)
        d["color_basis"] = np.log10(d["metric_raw"] + 1)
        metric_title = "Total Damage (USD)"
        metric_fmt = ":,.0f"
    else:
        d["metric_raw"] = d[metric_col].fillna(0).astype(float)
        d["color_basis"] = d["metric_raw"]
        metric_title = display_label(metric_col)
        metric_fmt = ":,.2f"
    return d, metric_title, metric_fmt

def make_prism(metric_col: str, grid_metric_col: str | None, start_month: str | None = None):
    d = STATE_MONTH.copy()
    d["lat"] = d["state_abbr"].map(lambda s: STATE_CENTROIDS.get(s, (np.nan, np.nan))[0])
    d["lon"] = d["state_abbr"].map(lambda s: STATE_CENTROIDS.get(s, (np.nan, np.nan))[1])
    d = d.dropna(subset=["lat", "lon"]).copy()

    d, metric_title, metric_fmt = _metric_preparation(d, metric_col)
    d["prism_bin"], _ = _assign_discrete_bins(d["color_basis"], n_bins=5)

    present_bins = [b for b in PRISM_ORDER if b in d["prism_bin"].unique().tolist()]
    months = sorted(d["month"].unique().tolist())
    initial_month = start_month if (start_month in months) else months[0]
    active_idx = months.index(initial_month)

    # Elevation tuning
    max_height_deg = 12.0
    v = d["metric_raw"].fillna(0).astype(float)
    v_max = float(v.max()) if float(v.max()) > 0 else 1.0
    HEIGHT_POWER = 0.42

    def spike_height(val: float) -> float:
        ratio = max(val, 0.0) / v_max
        return max_height_deg * (ratio ** HEIGHT_POWER)

    def cap_size(val: float) -> float:
        h = spike_height(val)
        return float(np.clip(6 + (h / max_height_deg) * 8, 6, 14))

    # normalize grid metric for outline width
    if grid_metric_col is not None and grid_metric_col in d.columns:
        gv = d[grid_metric_col].astype(float)
        finite = gv[np.isfinite(gv)]
        gmin = float(finite.min()) if len(finite) else 0.0
        gmax = float(finite.max()) if len(finite) else 1.0
    else:
        gmin, gmax = 0.0, 1.0

    def outline_width(grid_val: float) -> float:
        if grid_metric_col is None:
            return 1.0
        if not np.isfinite(grid_val) or gmax <= gmin:
            return 1.0
        t = (grid_val - gmin) / (gmax - gmin)
        return float(np.clip(1.0 + t * 3.0, 1.0, 4.0))

    def build_spike_lines(df_month, shadow=False):
        lons, lats = [], []
        for _, r in df_month.iterrows():
            h = spike_height(float(r["metric_raw"]))
            tilt = 0.22 if not shadow else 0.25
            lon0, lat0 = float(r["lon"]), float(r["lat"])
            lon1, lat1 = lon0 + tilt, lat0 + h
            lons += [lon0, lon1, None]
            lats += [lat0, lat1, None]
        return lons, lats

    def shadow_trace(df_month):
        lons, lats = build_spike_lines(df_month, shadow=True)
        return go.Scattergeo(
            lon=lons, lat=lats,
            mode="lines",
            line=dict(width=6, color="rgba(0,0,0,0.22)"),
            hoverinfo="skip",
            showlegend=False,
            name="shadow",
        )

    def spike_trace(df_bin, bin_name, showlegend=False):
        lons, lats = build_spike_lines(df_bin, shadow=False)
        return go.Scattergeo(
            lon=lons, lat=lats,
            mode="lines",
            line=dict(width=4, color=PRISM_COLORS[bin_name]),
            name=bin_name,
            legendgroup=bin_name,
            showlegend=showlegend,
            hoverinfo="skip",
        )

    def cap_trace(df_bin, bin_name):
        tilt = 0.22
        cap_lon, cap_lat, text, cd = [], [], [], []

        for _, r in df_bin.iterrows():
            h = spike_height(float(r["metric_raw"]))
            cap_lon.append(float(r["lon"]) + tilt)
            cap_lat.append(float(r["lat"]) + h)
            text.append(r["state_abbr"])

            gv = r.get(grid_metric_col, np.nan) if grid_metric_col else np.nan
            cd.append([
                r["month"], r["metric_raw"], r["storm_count"], r["severity"],
                r["fatalities"], r["total_damage_usd"], r.get("custom_region", None), gv
            ])

        cd = np.array(cd, dtype=object) if len(cd) else np.empty((0, 8), dtype=object)
        sizes = [cap_size(v) for v in df_bin["metric_raw"].fillna(0).astype(float)]

        if grid_metric_col is not None and grid_metric_col in df_bin.columns:
            outlines = [outline_width(float(x)) for x in df_bin[grid_metric_col].astype(float)]
        else:
            outlines = [1.0] * len(df_bin)

        hover_lines = [
            "<b>%{text}</b>",
            "Month: %{customdata[0]}",
            f"{metric_title}: <b>%{{customdata[1]{metric_fmt}}}</b>",
            "Storm Count: %{customdata[2]:,.0f}",
            "Severity: %{customdata[3]:,.1f}",
            "Fatalities: %{customdata[4]:,.0f}",
            "Total Damage: $%{customdata[5]:,.0f}",
            "EIA Region: %{customdata[6]}",
        ]
        if grid_metric_col is not None:
            #hover_lines.append(f"{display_label(grid_metric_col)}: %{customdata[7]:,.0f}")
            hover_lines.append(f"{display_label(grid_metric_col)}: %{{customdata[7]:,.0f}}")

        return go.Scattergeo(
            lon=cap_lon, lat=cap_lat,
            mode="markers",
            text=text,
            customdata=cd,
            marker=dict(
                size=sizes,
                color=PRISM_COLORS[bin_name],
                line=dict(width=outlines, color="rgba(255,255,255,0.85)"),
                opacity=0.95,
            ),
            showlegend=False,
            legendgroup=bin_name,
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            name=f"{bin_name} cap",
        )

    def build_month_data(dm):
        data = [shadow_trace(dm)]
        for b in present_bins:
            dfb = dm[dm["prism_bin"] == b].copy()
            data.append(spike_trace(dfb, b, showlegend=True))
            data.append(cap_trace(dfb, b))
        return data

    d0 = d[d["month"] == initial_month].copy()
    fig = go.Figure(data=build_month_data(d0))

    frames = []
    for m in months:
        dm = d[d["month"] == m].copy()
        frames.append(go.Frame(data=build_month_data(dm), name=m))
    fig.frames = frames

    slider_steps = [{
        "args": [[m], {"frame": {"duration": 600, "redraw": True},
                      "mode": "immediate",
                      "transition": {"duration": 200}}],
        "label": m,
        "method": "animate",
    } for m in months]

    fig.update_layout(
        title=dict(
            text=f"Prism spikes (discrete + depth) — {metric_title}",
            x=0.02, xanchor="left", y=0.99, yanchor="top",
            font=dict(size=18, color="#e5e7eb"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=0, r=0, t=105, b=125),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            bgcolor="rgba(0,0,0,0)",
            showland=True,
            landcolor="rgb(240, 243, 249)",
            subunitcolor="white",
            countrycolor="white",
            showlakes=False,
        ),
        legend=dict(
            title=None,
            orientation="h",
            x=0.02, xanchor="left",
            y=1.06, yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            itemwidth=38,
            tracegroupgap=6,
        ),
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0.02, "y": -0.10,
            "xanchor": "left", "yanchor": "bottom",
            "showactive": False,
            "buttons": [
                {"label": "▶", "method": "animate",
                 "args": [None, {"frame": {"duration": 600, "redraw": True},
                                 "fromcurrent": True,
                                 "transition": {"duration": 200}}]},
                {"label": "■", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": active_idx,
            "x": 0.10, "y": -0.10,
            "xanchor": "left", "yanchor": "bottom",
            "len": 0.86,
            "currentvalue": {"prefix": "Month: ", "font": {"color": "#e5e7eb"}},
            "pad": {"t": 18, "b": 0},
            "steps": slider_steps,
        }],
    )

    fig.add_annotation(
        text="Prism class:",
        x=0.02, y=1.075,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=11, color="#e5e7eb"),
    )

    return fig

# -------------------------
# Story beats list
# -------------------------
def outlier_list(n=8):
    o = outliers.sort_values("severity", ascending=False).head(n).copy()
    items = []
    for i, r in o.iterrows():
        label = f"{r['date'].date()} — {r['state_abbr']} | Severity: {r['severity']:.1f} | Storms: {int(r['storm_count'])}"
        items.append(
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
    return html.Ul(items, style={"listStyleType": "none", "paddingLeft": 0, "margin": 0})

# -------------------------
# Word cloud
# -------------------------
def make_wordcloud_image(freqs: dict):
    from wordcloud import WordCloud
    wc = WordCloud(width=1200, height=450, background_color=None, mode="RGBA", colormap="cividis")
    img = wc.generate_from_frequencies(freqs).to_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def wordcloud_or_bar(month_choice: str):
    if month_choice == "ALL":
        dfw = event_type_month.groupby("EVENT_TYPE", as_index=False)["storm_count"].sum()
    else:
        dfw = event_type_month[event_type_month["month"] == month_choice].groupby("EVENT_TYPE", as_index=False)["storm_count"].sum()

    dfw = dfw.sort_values("storm_count", ascending=False).head(60)
    freqs = dict(zip(dfw["EVENT_TYPE"].astype(str), dfw["storm_count"].astype(float)))
    uri = make_wordcloud_image(freqs)
    return html.Img(src=uri, style={"width": "100%", "borderRadius": "14px"})

# -------------------------
# Layout
# -------------------------
layout = html.Div([
    dcc.Store(id="selected-month", data=None),

    html.Div(className="card", children=[
        html.H3("Chapter 1 — Overview maps (2024)"),
        html.P(
            "NOAA shows storm intensity by state-month. EIA overlay adds grid stress by region-month (hover + outline strength).",
            className="small"
        ),

        html.Div(className="controls", children=[
            html.Div(className="control", children=[
                html.Label("View", className="small"),
                dcc.Dropdown(
                    id="map_view",
                    options=[
                        {"label": "Choropleth (animated)", "value": "choropleth"},
                        {"label": "Prism spikes (animated)", "value": "prism"},
                    ],
                    value="choropleth",
                    clearable=False,
                ),
            ]),
            html.Div(className="control", children=[
                html.Label("Storm metric (NOAA)", className="small"),
                dcc.Dropdown(
                    id="metric",
                    options=[{"label": k, "value": v} for k, v in METRICS.items()],
                    value="storm_count",
                    clearable=False,
                ),
            ]),
            html.Div(className="control", children=[
                html.Label("Grid overlay (EIA)", className="small"),
                dcc.Dropdown(
                    id="grid_metric",
                    options=[{"label": k, "value": v} for k, v in GRID_METRICS.items()],
                    value="mean_abs_anomaly_mw",
                    clearable=False,
                ),
            ]),
        ]),

        dcc.Graph(id="overview_map", style={"height": "660px"}, config={"displayModeBar": False}),
        html.P("Tip: use the outlier list below to jump to a case-study month.", className="small"),
    ]),

    html.Div(className="card", children=[
        html.H3("Story beats — top extreme state-days (click to jump)"),
        html.P("Click to jump map animation to that month (case study).", className="small"),
        html.Div(children=outlier_list(8)),
        dcc.Markdown(id="outlier-selected", className="small", style={"marginTop": "10px"})
    ]),

    html.Div(className="card", children=[
        html.H3("Word cloud — storm event types"),
        html.P("Size encodes event frequency. Select a month to see seasonality.", className="small"),
        html.Div(className="controls", children=[
            html.Div(className="control", children=[
                html.Label("Month", className="small"),
                dcc.Dropdown(
                    id="wc_month",
                    options=[{"label": m, "value": m} for m in WC_MONTHS],
                    value="ALL",
                    clearable=False,
                ),
            ]),
        ]),
        html.Div(id="wc_container"),
    ]),
])

# -------------------------
# Callbacks
# -------------------------
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
    msg = (
        f"**Case study selected** → {row['date'].date()} ({month}) in **{row['state_abbr']}** | "
        f"Storms: {int(row['storm_count'])} | Fatalities: {int(row['fatalities'])} | "
        f"Damage: ${row['total_damage_usd']:,.0f} | Severity: {row['severity']:.1f}"
    )
    return month, msg

@dash.callback(
    Output("overview_map", "figure"),
    Input("map_view", "value"),
    Input("metric", "value"),
    Input("grid_metric", "value"),
    Input("selected-month", "data"),
)
def update_map(map_view, metric_col, grid_metric_col, selected_month):
    # ✅ Convert sentinel to None for internal logic
    if grid_metric_col == "none":
        grid_metric_col = None

    if map_view == "prism":
        return make_prism(metric_col, grid_metric_col, selected_month)
    return make_choropleth(metric_col, grid_metric_col, selected_month)

@dash.callback(
    Output("wc_container", "children"),
    Input("wc_month", "value"),
)
def update_wc(month_choice):
    return wordcloud_or_bar(month_choice)