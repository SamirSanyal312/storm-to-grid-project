import argparse
import glob
import os
import re
import pandas as pd
import numpy as np

STATE_ABBR = {
    'ALABAMA':'AL','ALASKA':'AK','ARIZONA':'AZ','ARKANSAS':'AR','CALIFORNIA':'CA','COLORADO':'CO',
    'CONNECTICUT':'CT','DELAWARE':'DE','DISTRICT OF COLUMBIA':'DC','FLORIDA':'FL','GEORGIA':'GA',
    'HAWAII':'HI','IDAHO':'ID','ILLINOIS':'IL','INDIANA':'IN','IOWA':'IA','KANSAS':'KS','KENTUCKY':'KY',
    'LOUISIANA':'LA','MAINE':'ME','MARYLAND':'MD','MASSACHUSETTS':'MA','MICHIGAN':'MI','MINNESOTA':'MN',
    'MISSISSIPPI':'MS','MISSOURI':'MO','MONTANA':'MT','NEBRASKA':'NE','NEVADA':'NV','NEW HAMPSHIRE':'NH',
    'NEW JERSEY':'NJ','NEW MEXICO':'NM','NEW YORK':'NY','NORTH CAROLINA':'NC','NORTH DAKOTA':'ND','OHIO':'OH',
    'OKLAHOMA':'OK','OREGON':'OR','PENNSYLVANIA':'PA','RHODE ISLAND':'RI','SOUTH CAROLINA':'SC',
    'SOUTH DAKOTA':'SD','TENNESSEE':'TN','TEXAS':'TX','UTAH':'UT','VERMONT':'VT','VIRGINIA':'VA',
    'WASHINGTON':'WA','WEST VIRGINIA':'WV','WISCONSIN':'WI','WYOMING':'WY'
}

def parse_damage(x: str) -> float:
    """NOAA DAMAGE_PROPERTY/DAMAGE_CROPS like '10.00K', '2.5M' -> dollars."""
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s == "" or s.upper() in {"N/A", "NA"}:
        return 0.0
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMB])?\s*$", s, flags=re.I)
    if not m:
        return 0.0
    val = float(m.group(1))
    suf = (m.group(2) or "").upper()
    mult = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}.get(suf, 1)
    return val * mult

def load_state_to_region(map_path: str) -> pd.DataFrame:
    mapping = pd.read_csv(map_path)
    if not {"state_abbr", "region"}.issubset(set(mapping.columns)):
        raise ValueError(f"{map_path} must contain columns: state_abbr, region")

    mapping = mapping.dropna(subset=["state_abbr", "region"]).copy()
    mapping["state_abbr"] = mapping["state_abbr"].astype(str).str.upper().str.strip()
    mapping["region"] = mapping["region"].astype(str).str.strip()
    mapping = mapping[mapping["state_abbr"].str.len() > 0].copy()

    dup = mapping["state_abbr"].duplicated().sum()
    if dup > 0:
        dups = mapping[mapping["state_abbr"].duplicated(keep=False)].sort_values("state_abbr")
        raise ValueError(f"{map_path} has duplicate state_abbr entries.\n\n{dups}")

    return mapping

def load_eia(eia_glob: str, year: int) -> pd.DataFrame:
    files = sorted(glob.glob(eia_glob))
    if not files:
        raise FileNotFoundError(f"No EIA files match: {eia_glob}")

    usecols = [
        "Balancing Authority", "Data Date", "Hour Number",
        "Demand Forecast (MW)", "Demand (MW)",
        "Net Generation (MW)", "Total Interchange (MW)", "Region"
    ]
    df = pd.concat([pd.read_csv(f, usecols=usecols, low_memory=False) for f in files], ignore_index=True)

    # Robust date parsing for EIA (handles 01-01-2024 and 01/01/2024)
    raw = df["Data Date"].astype(str).str.strip()
    d1 = pd.to_datetime(raw, format="%m-%d-%Y", errors="coerce")
    d2 = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
    df["Data Date"] = d1.fillna(d2)
    df.loc[df["Data Date"].isna(), "Data Date"] = pd.to_datetime(raw, errors="coerce")

    df = df[df["Data Date"].dt.year == year].copy()

    for c in ["Demand Forecast (MW)", "Demand (MW)", "Net Generation (MW)", "Total Interchange (MW)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Demand Anomaly (MW)"] = df["Demand (MW)"] - df["Demand Forecast (MW)"]
    df["Region"] = df["Region"].astype(str).str.upper().str.strip()
    df["Hour Number"] = pd.to_numeric(df["Hour Number"], errors="coerce")

    df = df.dropna(subset=["Region", "Data Date", "Hour Number"])

    if df.empty:
        sample = raw.head(10).tolist()
        raise ValueError(
            "EIA data became empty after parsing/filtering.\n"
            f"Sample Data Date values: {sample}"
        )

    print("EIA sample dates:", df["Data Date"].head(3).tolist())
    print("EIA unique regions:", sorted(df["Region"].dropna().unique().tolist()))
    return df

def load_storm_details(details_path: str, year: int) -> pd.DataFrame:
    usecols = [
        "EVENT_ID", "STATE", "YEAR", "EVENT_TYPE",
        "BEGIN_DATE_TIME", "END_DATE_TIME",
        "DAMAGE_PROPERTY", "DAMAGE_CROPS",
        "INJURIES_DIRECT", "DEATHS_DIRECT"
    ]
    df = pd.read_csv(details_path, usecols=usecols, low_memory=False)
    df = df[df["YEAR"] == year].copy()

    # NOAA timestamps in your file are typically day-first
    df["BEGIN_DATE_TIME"] = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce", dayfirst=True)
    df["END_DATE_TIME"] = pd.to_datetime(df["END_DATE_TIME"], errors="coerce", dayfirst=True)

    df["STATE"] = df["STATE"].astype(str).str.upper().str.strip()
    df["state_abbr"] = df["STATE"].map(STATE_ABBR)

    df["damage_property_usd"] = df["DAMAGE_PROPERTY"].apply(parse_damage)
    df["damage_crops_usd"] = df["DAMAGE_CROPS"].apply(parse_damage)

    df["inj_direct"] = pd.to_numeric(df["INJURIES_DIRECT"], errors="coerce").fillna(0)
    df["death_direct"] = pd.to_numeric(df["DEATHS_DIRECT"], errors="coerce").fillna(0)

    # Unitless severity score for ranking/storytelling
    df["severity"] = (
        np.log1p(df["damage_property_usd"] + df["damage_crops_usd"])
        + 5.0 * df["death_direct"]
        + 1.0 * df["inj_direct"]
    )

    df["date"] = df["BEGIN_DATE_TIME"].dt.floor("D")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df

def load_fatalities(fatalities_path: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(fatalities_path, low_memory=False)
    df["year"] = pd.to_numeric(df["FAT_YEARMONTH"].astype(str).str[:4], errors="coerce")
    df = df[df["year"] == year].copy()

    fat_by_event = df.groupby("EVENT_ID", as_index=False).agg(
        fatalities_count=("FATALITY_ID", "count")
    )
    return fat_by_event

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--eia_glob", default="data/raw/EIA930_BALANCE_2024_*.csv")
    ap.add_argument("--storms_details", required=True)
    ap.add_argument("--storms_fatalities", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--state_region_map", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- Load data ----------------
    print("Loading EIA…")
    eia = load_eia(args.eia_glob, args.year)

    print("Loading NOAA storm details…")
    storms = load_storm_details(args.storms_details, args.year)

    print("Loading NOAA fatalities…")
    fat_by_event = load_fatalities(args.storms_fatalities, args.year)

    storms = storms.merge(fat_by_event, on="EVENT_ID", how="left")
    storms["fatalities_count"] = storms["fatalities_count"].fillna(0).astype(int)

    # ---------------- EIA (keep region_daily.pkl for reference/other pages) ----------------
    region_daily = eia.groupby(["Region", "Data Date"], as_index=False).agg({
        "Demand (MW)": "sum",
        "Demand Forecast (MW)": "sum",
        "Demand Anomaly (MW)": "sum",
        "Net Generation (MW)": "sum",
        "Total Interchange (MW)": "sum",
    }).rename(columns={"Data Date": "date"})

    # ---------------- EIA: CORRECT grid-stress computation ----------------
    # Step 1: Hourly regional anomaly (sum across BAs per hour)
    eia_hourly = eia.groupby(["Region", "Data Date", "Hour Number"], as_index=False).agg({
        "Demand Anomaly (MW)": "sum"
    }).rename(columns={"Data Date": "date", "Hour Number": "hour"})

    # Step 2: Daily stress per EIA region (mean abs across 24 hours)
    eia_hourly["abs_anom"] = eia_hourly["Demand Anomaly (MW)"].abs()

    eia_region_daily_stress = eia_hourly.groupby(["Region", "date"], as_index=False).agg(
        anomaly_abs_mean_mw=("abs_anom", "mean"),
        anomaly_abs_max_mw=("abs_anom", "max"),
        anomaly_mean_mw=("Demand Anomaly (MW)", "mean"),
    )

    # Step 3: Map EIA regions -> your custom regions (matches your actual labels)
    EIA_TO_CUSTOM = {
        # Northeast family
        "NE": "NE",
        "NY": "NE",
        "MIDA": "NE",

        # Southeast family
        "SE": "SE",
        "CAR": "SE",
        "FLA": "SE",
        "TEN": "SE",

        # Others
        "MIDW": "MIDW",
        "CENT": "PLAINS",
        "SW": "SW",
        "NW": "NW",
        "CAL": "PAC",
        "TEX": "TX",
    }

    eia_region_daily_stress["Region"] = eia_region_daily_stress["Region"].astype(str).str.upper().str.strip()
    eia_region_daily_stress["custom_region"] = eia_region_daily_stress["Region"].map(EIA_TO_CUSTOM)

    unmapped_eia = sorted(
        eia_region_daily_stress.loc[eia_region_daily_stress["custom_region"].isna(), "Region"]
        .unique()
        .tolist()
    )
    if unmapped_eia:
        print("⚠️ Unmapped EIA Region labels (update EIA_TO_CUSTOM if needed):", unmapped_eia)

    # Step 4: Aggregate into your custom regions per day
    eia_custom_region_daily = (
        eia_region_daily_stress.dropna(subset=["custom_region"])
        .groupby(["custom_region", "date"], as_index=False)
        .agg(
            anomaly_abs_mean_mw=("anomaly_abs_mean_mw", "mean"),
            anomaly_abs_max_mw=("anomaly_abs_max_mw", "max"),
            anomaly_mean_mw=("anomaly_mean_mw", "mean"),
        )
    )

    if eia_custom_region_daily.empty:
        raise ValueError(
            "eia_custom_region_daily became empty. Mapping did not match EIA regions."
        )

    # ---------------- Storm daily per state ----------------
    storms_daily_state = storms.groupby(["state_abbr", "date"], as_index=False).agg(
        storm_count=("EVENT_ID", "count"),
        severity=("severity", "sum"),
        total_damage_usd=("damage_property_usd", "sum"),
        fatalities=("fatalities_count", "sum"),
    )
    storms_daily_state["month"] = storms_daily_state["date"].dt.to_period("M").astype(str)

    # Monthly state choropleth
    state_month = storms_daily_state.groupby(["state_abbr", "month"], as_index=False).agg(
        storm_count=("storm_count", "sum"),
        severity=("severity", "sum"),
        total_damage_usd=("total_damage_usd", "sum"),
        fatalities=("fatalities", "sum"),
    )

    # Event type seasonality
    event_type_month = storms.groupby(["EVENT_TYPE", "month"], as_index=False).agg(
        storm_count=("EVENT_ID", "count"),
        severity=("severity", "sum"),
        fatalities=("fatalities_count", "sum"),
    )

    # Outliers
    outliers_state_day = storms_daily_state.sort_values("severity", ascending=False).head(25)

    # ---------------- Storm daily -> custom region via state_to_region.csv ----------------
    map_path = args.state_region_map or os.path.join(args.out_dir, "state_to_region.csv")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Missing mapping file: {map_path}")

    mapping = load_state_to_region(map_path)
    storms_state_with_region = storms_daily_state.merge(mapping, on="state_abbr", how="left")

    unmapped_storms = storms_state_with_region["region"].isna().sum()
    if unmapped_storms > 0:
        print(f"⚠️ Warning: {unmapped_storms} storm rows have no custom region mapping.")

    storms_region_daily = storms_state_with_region.groupby(["region", "date"], as_index=False).agg(
        storm_count=("storm_count", "sum"),
        severity=("severity", "sum"),
        total_damage_usd=("total_damage_usd", "sum"),
        fatalities=("fatalities", "sum"),
    )

    # ---------------- Save outputs ----------------
    state_month.to_pickle(os.path.join(args.out_dir, "state_month.pkl"))
    region_daily.to_pickle(os.path.join(args.out_dir, "region_daily.pkl"))
    eia_custom_region_daily.to_pickle(os.path.join(args.out_dir, "eia_custom_region_daily.pkl"))
    event_type_month.to_pickle(os.path.join(args.out_dir, "event_type_month.pkl"))
    outliers_state_day.to_pickle(os.path.join(args.out_dir, "outliers_state_day.pkl"))
    storms_region_daily.to_pickle(os.path.join(args.out_dir, "storms_region_daily.pkl"))

    print("✅ Done. Wrote:")
    print(" - state_month.pkl")
    print(" - region_daily.pkl")
    print(" - eia_custom_region_daily.pkl (hourly->daily stress: mean/max abs + mean)")
    print(" - event_type_month.pkl")
    print(" - outliers_state_day.pkl")
    print(" - storms_region_daily.pkl")

if __name__ == "__main__":
    main()