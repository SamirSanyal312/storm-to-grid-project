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

    df["Data Date"] = pd.to_datetime(df["Data Date"], format="%m-%d-%Y", errors="coerce")
    df = df[df["Data Date"].dt.year == year].copy()

    for c in ["Demand Forecast (MW)", "Demand (MW)", "Net Generation (MW)", "Total Interchange (MW)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Demand Anomaly (MW)"] = df["Demand (MW)"] - df["Demand Forecast (MW)"]
    df = df.dropna(subset=["Region", "Data Date", "Hour Number"])
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

    # dayfirst=True because your NOAA file uses dd-mm-yyyy style timestamps
    df["BEGIN_DATE_TIME"] = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce", dayfirst=True)
    df["END_DATE_TIME"] = pd.to_datetime(df["END_DATE_TIME"], errors="coerce", dayfirst=True)

    df["STATE"] = df["STATE"].astype(str).str.upper().str.strip()
    df["state_abbr"] = df["STATE"].map(STATE_ABBR)

    df["damage_property_usd"] = df["DAMAGE_PROPERTY"].apply(parse_damage)
    df["damage_crops_usd"] = df["DAMAGE_CROPS"].apply(parse_damage)

    df["inj_direct"] = pd.to_numeric(df["INJURIES_DIRECT"], errors="coerce").fillna(0)
    df["death_direct"] = pd.to_numeric(df["DEATHS_DIRECT"], errors="coerce").fillna(0)

    # Unitless severity index for ranking
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

def load_state_to_region(map_path: str) -> pd.DataFrame:
    mapping = pd.read_csv(map_path)

    if not {"state_abbr", "region"}.issubset(set(mapping.columns)):
        raise ValueError(f"{map_path} must contain columns: state_abbr, region")

    # Drop blank rows safely (this is what fixes your error)
    mapping = mapping.dropna(subset=["state_abbr", "region"]).copy()
    mapping["state_abbr"] = mapping["state_abbr"].astype(str).str.upper().str.strip()
    mapping["region"] = mapping["region"].astype(str).str.strip()

    # Remove rows where state_abbr is empty string after stripping
    mapping = mapping[mapping["state_abbr"].str.len() > 0].copy()

    # Ensure one-to-one mapping
    dup = mapping["state_abbr"].duplicated().sum()
    if dup > 0:
        dups = mapping[mapping["state_abbr"].duplicated(keep=False)].sort_values("state_abbr")
        raise ValueError(
            f"{map_path} has duplicate state_abbr entries. Fix duplicates first.\n\n{dups}"
        )

    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--eia_glob", default="data/raw/EIA930_BALANCE_2024_*.csv")
    ap.add_argument("--storms_details", required=True)
    ap.add_argument("--storms_fatalities", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--state_region_map", default=None, help="Optional path to state_to_region.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading EIA…")
    eia = load_eia(args.eia_glob, args.year)

    print("Loading NOAA storm details…")
    storms = load_storm_details(args.storms_details, args.year)

    print("Loading NOAA fatalities…")
    fat_by_event = load_fatalities(args.storms_fatalities, args.year)

    storms = storms.merge(fat_by_event, on="EVENT_ID", how="left")
    storms["fatalities_count"] = storms["fatalities_count"].fillna(0).astype(int)

    # ---- EIA daily ----
    eia_daily = eia.groupby(["Region", "Data Date"], as_index=False).agg({
        "Demand (MW)": "sum",
        "Demand Forecast (MW)": "sum",
        "Demand Anomaly (MW)": "sum",
        "Net Generation (MW)": "sum",
        "Total Interchange (MW)": "sum",
    }).rename(columns={"Data Date": "date"})

    # ---- Storm daily per state ----
    storms_daily_state = storms.groupby(["state_abbr", "date"], as_index=False).agg(
        storm_count=("EVENT_ID", "count"),
        severity=("severity", "sum"),
        total_damage_usd=("damage_property_usd", "sum"),
        fatalities=("fatalities_count", "sum"),
    )
    storms_daily_state["month"] = storms_daily_state["date"].dt.to_period("M").astype(str)

    # ---- Monthly state choropleth ----
    state_month = storms_daily_state.groupby(["state_abbr", "month"], as_index=False).agg(
        storm_count=("storm_count", "sum"),
        severity=("severity", "sum"),
        total_damage_usd=("total_damage_usd", "sum"),
        fatalities=("fatalities", "sum"),
    )

    # ---- Event type seasonality ----
    event_type_month = storms.groupby(["EVENT_TYPE", "month"], as_index=False).agg(
        storm_count=("EVENT_ID", "count"),
        severity=("severity", "sum"),
        fatalities=("fatalities_count", "sum"),
    )

    # ---- Outliers ----
    outliers_state_day = storms_daily_state.sort_values("severity", ascending=False).head(25)

    # ---- Custom region aggregation ----
    map_path = args.state_region_map or os.path.join(args.out_dir, "state_to_region.csv")
    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"Missing mapping file: {map_path}\nCreate it first with columns: state_abbr,region"
        )

    mapping = load_state_to_region(map_path)
    storms_state_with_region = storms_daily_state.merge(mapping, on="state_abbr", how="left")

    unmapped = storms_state_with_region["region"].isna().sum()
    if unmapped > 0:
        print(f"⚠️ Warning: {unmapped} state-day rows have no region mapping (check {map_path}).")

    storms_region_daily = storms_state_with_region.groupby(["region", "date"], as_index=False).agg(
        storm_count=("storm_count", "sum"),
        severity=("severity", "sum"),
        total_damage_usd=("total_damage_usd", "sum"),
        fatalities=("fatalities", "sum"),
    )

    # ---- Save outputs ----
    state_month.to_pickle(os.path.join(args.out_dir, "state_month.pkl"))
    eia_daily.to_pickle(os.path.join(args.out_dir, "region_daily.pkl"))
    event_type_month.to_pickle(os.path.join(args.out_dir, "event_type_month.pkl"))
    outliers_state_day.to_pickle(os.path.join(args.out_dir, "outliers_state_day.pkl"))
    storms_region_daily.to_pickle(os.path.join(args.out_dir, "storms_region_daily.pkl"))

    print("✅ Done. Wrote:")
    print(" - state_month.pkl")
    print(" - region_daily.pkl")
    print(" - event_type_month.pkl")
    print(" - outliers_state_day.pkl")
    print(" - storms_region_daily.pkl")

if __name__ == "__main__":
    main()