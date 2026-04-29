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

outliers = pd.read_pickle("data/processed/outliers_state_day.pkl").copy()
outliers["date"] = pd.to_datetime(outliers["date"], errors="coerce")
outliers["month"] = outliers["date"].dt.to_period("M").astype(str)  # use "M"

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

# Keep values as strings (avoid None/null edge cases in Dropdown options)
GRID_METRICS = {
    "None (NOAA only)": "__none__",
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

def info_tip(text: str):
    return html.Span("i", className="info", **{"data-tip": text})

WC_MONTHS = ["ALL"] + sorted(event_type_month["month"].unique().tolist())

# Approx state centroids for prism spikes
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
    # We'll swap to a colorblind-friendly scale later (you mentioned it)
    return "cividis"

def _set_slider_to_month(fig, month_value: str | None, months: list[str]):
    if not month_value or not fig.layout.sliders:
        return
    if month_value in months:
        fig.layout.sliders[0].active = months.index(month_value)

# -------------------------
# EIA join: state-month + region-month grid metrics
# -------------------------
def load_state_to_region():
    m = pd.read_csv(STATE_TO_REGION_PATH)
    m.columns = [c.strip() for c in m.columns]

    state_candidates = ["state_abbr", "state", "abbr", "STATE", "State", "STATE_ABBR"]
    state_col = next((c for c in state_candidates if c in m.columns), None)
    if state_col is None:
        raise ValueError(
            f"state_to_region.csv must contain a state column like 'state_abbr'. Found: {m.columns.tolist()}"
        )

    region_candidates = [
        "custom_region", "region", "Region", "EIA Region", "EIA_Region",
        "eia_region", "eiaRegion", "customRegion", "CUSTOM_REGION"
    ]
    region_col = next((c for c in region_candidates if c in m.columns), None)
    if region_col is None:
        other_cols = [c for c in m.columns if c != state_col]
        if len(other_cols) == 1:
            region_col = other_cols[0]
        else:
            raise ValueError(
                f"Couldn't infer region column in state_to_region.csv. Found columns: {m.columns.tolist()}"
            )

    m = m.rename(columns={state_col: "state_abbr", region_col: "custom_region"})
    m["state_abbr"] = m["state_abbr"].astype(str).str.strip().str.upper()
    m["custom_region"] = m["custom_region"].astype(str).str.strip()
    return m[["state_abbr", "custom_region"]]

def _normalize_colname(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace("(", "").replace(")", "")
    c = c.replace("|", " ").replace("/", " ")
    c = " ".join(c.split())
    return c

def load_region_month_grid():
    """
    Reads eia_custom_region_daily.pkl and produces region-month aggregates of:
      - mean_abs_anomaly_mw
      - max_abs_anomaly_mw
      - signed_mean_anomaly_mw

    IMPORTANT: supports either:
      anomaly_abs_mean_mw / anomaly_abs_max_mw / anomaly_mean_mw
    OR:
      anomaly abs mean mw / anomaly abs max mw / anomaly mean mw
    """
    df = pd.read_pickle(EIA_DAILY_PATH).copy()

    # unify date column
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # unify region column
    if "custom_region" not in df.columns and "Region" in df.columns:
        df = df.rename(columns={"Region": "custom_region"})
    if "custom_region" not in df.columns:
        raise ValueError(f"EIA file is missing custom_region/Region. Columns: {df.columns.tolist()}")
    df["custom_region"] = df["custom_region"].astype(str).str.strip()

    # month
    if "month" not in df.columns:
        df["month"] = df["date"].dt.to_period("M").astype(str)
    else:
        df["month"] = df["month"].astype(str)

    # Normalize column names into a lookup to make renaming robust
    raw_cols = list(df.columns)
    norm_map = {_normalize_colname(c): c for c in raw_cols}

    # candidates (normalized -> canonical)
    want = {
        "anomaly abs mean mw": "mean_abs_anomaly_mw",
        "anomaly_abs_mean_mw": "mean_abs_anomaly_mw",

        "anomaly abs max mw": "max_abs_anomaly_mw",
        "anomaly_abs_max_mw": "max_abs_anomaly_mw",

        "anomaly mean mw": "signed_mean_anomaly_mw",
        "anomaly_mean_mw": "signed_mean_anomaly_mw",
    }

    rename_real = {}
    for k_norm, canonical in want.items():
        if k_norm in norm_map:
            rename_real[norm_map[k_norm]] = canonical

    df = df.rename(columns=rename_real)

    agg_cols = [c for c in ["mean_abs_anomaly_mw", "max_abs_anomaly_mw", "signed_mean_anomaly_mw"] if c in df.columns]
    if not agg_cols:
        raise ValueError(
            "Could not find expected EIA anomaly columns after normalization/renaming. "
            f"Columns seen: {sorted(df.columns.tolist())}"
        )

    # region-month mean (you can switch to median later if you want a more robust summary)
    g = df.groupby(["custom_region", "month"], as_index=False)[agg_cols].mean()
    return g

def build_state_month_enriched():
    base = state_month_raw.copy()
    base["state_abbr"] = base["state_abbr"].astype(str).str.upper()
    s2r = load_state_to_region()
    rmonth = load_region_month_grid()
    base = base.merge(s2r, on="state_abbr", how="left")
    base = base.merge(rmonth, on=["custom_region", "month"], how="left")
    return base

STATE_MONTH = build_state_month_enriched()

# -------------------------
# Choropleth
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
        "severity": ":,.1f",
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
# Prism spikes (discrete + depth)
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

    # Stronger elevation (you asked “increase elevation”)
    max_height_deg = 18.0
    v = d["metric_raw"].fillna(0).astype(float)
    v_max = float(v.max()) if float(v.max()) > 0 else 1.0
    HEIGHT_POWER = 0.38

    def spike_height(val: float) -> float:
        ratio = max(val, 0.0) / v_max
        return max_height_deg * (ratio ** HEIGHT_POWER)

    def cap_size(val: float) -> float:
        h = spike_height(val)
        return float(np.clip(7 + (h / max_height_deg) * 10, 7, 18))

    # Outline width encodes grid overlay magnitude (if selected)
    if grid_metric_col is not None and grid_metric_col in d.columns:
        gv = d[grid_metric_col].astype(float)
        gmin = np.nanmin(gv.values) if np.isfinite(np.nanmin(gv.values)) else 0.0
        gmax = np.nanmax(gv.values) if np.isfinite(np.nanmax(gv.values)) else 1.0
    else:
        gmin, gmax = 0.0, 1.0

    def outline_width(grid_val: float) -> float:
        if grid_metric_col is None:
            return 1.0
        if not np.isfinite(grid_val):
            return 1.0
        if gmax <= gmin:
            return 1.0
        t = (grid_val - gmin) / (gmax - gmin)
        return float(np.clip(1.0 + t * 3.2, 1.0, 4.2))

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
            line=dict(width=7, color="rgba(0,0,0,0.24)"),
            hoverinfo="skip",
            showlegend=False,
        )

    def spike_trace(df_bin, bin_name, showlegend=False):
        lons, lats = build_spike_lines(df_bin, shadow=False)
        return go.Scattergeo(
            lon=lons, lat=lats,
            mode="lines",
            line=dict(width=4.8, color=PRISM_COLORS[bin_name]),
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
            gv = r.get(grid_metric_col, np.nan) if (grid_metric_col and grid_metric_col in r) else np.nan
            cd.append([
                r["month"], r["metric_raw"], r["storm_count"], r["severity"],
                r["fatalities"], r["total_damage_usd"], r.get("custom_region", None), gv
            ])

        cd = np.array(cd, dtype=object) if len(cd) else np.empty((0, 8), dtype=object)
        sizes = [cap_size(v) for v in df_bin["metric_raw"].fillna(0).astype(float)]

        outlines = []
        if grid_metric_col and grid_metric_col in df_bin.columns:
            outlines = [outline_width(float(x)) for x in pd.to_numeric(df_bin[grid_metric_col], errors="coerce")]
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
        if grid_metric_col and grid_metric_col in df_bin.columns:
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
                line=dict(width=outlines, color="rgba(255,255,255,0.88)"),
                opacity=0.96,
            ),
            showlegend=False,
            legendgroup=bin_name,
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
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
                      "transition": {"duration": 220}}],
        "label": m,
        "method": "animate",
    } for m in months]

    fig.update_layout(
        title=dict(
            text=f"Prism spikes (discrete + depth) — {metric_title}",
            x=0.02, xanchor="left",
            y=0.99, yanchor="top",
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
            landcolor="rgb(242, 245, 250)",
            subunitcolor="white",
            countrycolor="white",
            showlakes=False,
        ),
        legend=dict(
            orientation="h",
            x=0.06, xanchor="left",
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
                                 "transition": {"duration": 220}}]},
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
                    className="story-btn"
                )
            )
        )
    return html.Ul(items, className="story-list")

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
    return html.Img(src=uri, className="wc-img")

# -------------------------
# Layout (Premium)
# -------------------------
layout = html.Div([
    dcc.Store(id="selected-month", data=None),

    html.Div(className="card", children=[
        html.Div(className="section-head", children=[
            html.Div(children=[
                html.H3("Chapter 1 — Overview maps (2024)", style={"margin": 0}),
                html.Div(className="small", children=[
                    "NOAA shows storm intensity by state-month. EIA overlay adds grid stress by region-month (hover + outline strength)."
                ]),
            ]),
            html.Div(className="badge-row", children=[
                html.Span("2024 scope", className="badge"),
                html.Span("NOAA Storm Events", className="badge badge-2"),
                html.Span("EIA-930 anomalies", className="badge badge-3"),
            ])
        ]),

        html.Div(className="controls", children=[
            html.Div(className="control", children=[
                html.Div(className="label-row", children=[
                    html.Label("View", className="small"),
                    info_tip("Choropleth = geographic context. Prism spikes = discrete bins + elevation encoding for fast comparison.")
                ]),
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
                html.Div(className="label-row", children=[
                    html.Label("Storm metric (NOAA)", className="small"),
                    info_tip("Count = frequency. Severity = composite intensity. Damage/Fatalities = impact.")
                ]),
                dcc.Dropdown(
                    id="metric",
                    options=[{"label": k, "value": v} for k, v in METRICS.items()],
                    value="storm_count",
                    clearable=False,
                ),
            ]),
            html.Div(className="control", children=[
                html.Div(className="label-row", children=[
                    html.Label("Grid overlay (EIA)", className="small"),
                    info_tip("Hover a state to see its EIA region + overlay value. Prism outlines get thicker when grid stress is higher.")
                ]),
                dcc.Dropdown(
                    id="grid_metric",
                    options=[{"label": k, "value": v} for k, v in GRID_METRICS.items()],
                    value="mean_abs_anomaly_mw",
                    clearable=False,
                ),
            ]),
        ]),

        html.Div(id="overview_kpis", className="kpis"),

        html.Div(className="chart-card", children=[
            html.Div(className="chart-title", children="Main view"),
            html.Div(className="chart-subtitle", children=[
                "Tip: watch month-by-month patterns. Then click Story beats to jump to extreme periods."
            ]),
            dcc.Graph(id="overview_map", style={"height": "660px"}, config={"displayModeBar": False}),
            html.Div(className="callout", children=[
                html.Div(className="small", children=[
                    html.B("Takeaway: "),
                    "NOAA shows where storms concentrate. EIA overlay contextualizes whether those months also show elevated grid stress (exploratory)."
                ])
            ])
        ]),
    ]),

    html.Div(className="card", children=[
        html.Div(className="section-head", children=[
            html.H3("Story beats — top extreme state-days (click to jump)"),
            html.Div(className="small", children="Details-on-demand: click a state-day to jump the map to its month and show a case study summary.")
        ]),
        html.Div(children=outlier_list(8)),
        dcc.Markdown(id="outlier-selected", className="small", style={"marginTop": "10px"})
    ]),

    html.Div(className="card", children=[
        html.Div(className="section-head", children=[
            html.H3("Word cloud — storm event types"),
            html.Div(className="small", children="Size encodes event frequency. Select a month to see seasonal shifts.")
        ]),
        html.Div(className="controls wc-controls", children=[
            html.Div(className="control", children=[
                html.Div(className="label-row", children=[
                    html.Label("Month", className="small"),
                    info_tip("ALL shows overall distribution. A single month highlights seasonal changes (winter vs summer event types).")
                ]),
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
def update_map(map_view, metric_col, grid_metric_key, selected_month):
    grid_metric_col = None if grid_metric_key == "__none__" else grid_metric_key
    if map_view == "prism":
        return make_prism(metric_col, grid_metric_col, selected_month)
    return make_choropleth(metric_col, grid_metric_col, selected_month)

@dash.callback(
    Output("wc_container", "children"),
    Input("wc_month", "value"),
)
def update_wc(month_choice):
    return wordcloud_or_bar(month_choice)

@dash.callback(
    Output("overview_kpis", "children"),
    Input("selected-month", "data"),
    Input("grid_metric", "value"),
)
def update_overview_kpis(selected_month, grid_metric_key):
    grid_metric_col = None if grid_metric_key == "__none__" else grid_metric_key

    d = STATE_MONTH.copy()
    scope_label = "All months (2024)"
    if selected_month and selected_month in d["month"].unique():
        d = d[d["month"] == selected_month].copy()
        scope_label = f"Month: {selected_month}"

    total_storms = float(d["storm_count"].fillna(0).sum())
    total_fatal = float(d["fatalities"].fillna(0).sum())
    total_damage = float(d["total_damage_usd"].fillna(0).sum())

    grid_summary = "—"
    if grid_metric_col and grid_metric_col in d.columns:
        vals = pd.to_numeric(d[grid_metric_col], errors="coerce")
        if vals.notna().any():
            grid_summary = f"{vals.mean():,.0f} MW (mean across states’ regions)"

    return [
        html.Div(className="kpi", children=[
            html.Div(className="label", children=f"Total Storms ({scope_label})"),
            html.Div(className="value", children=f"{total_storms:,.0f}"),
        ]),
        html.Div(className="kpi", children=[
            html.Div(className="label", children=f"Total Fatalities ({scope_label})"),
            html.Div(className="value", children=f"{total_fatal:,.0f}"),
        ]),
        html.Div(className="kpi", children=[
            html.Div(className="label", children=f"Total Damage ({scope_label})"),
            html.Div(className="value", children=f"${total_damage:,.0f}"),
            html.Div(className="small", style={"marginTop": "6px"}, children=[
                html.B("Grid overlay summary: "), grid_summary
            ])
        ]),
    ]