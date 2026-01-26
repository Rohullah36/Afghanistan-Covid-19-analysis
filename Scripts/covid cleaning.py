#!/usr/bin/env python3
"""
clean_data.py

Edit the INPUT / OUTPUT filenames below if needed and run:
    python clean_data.py
"""
import pandas as pd
import numpy as np
import re

# Input / output
INPUT = "data.csv"
OUTPUT = "cleaned-data.csv"

# Read the data
df = pd.read_csv(INPUT)

# Drop first row safely (if you really want to drop the top data row)
# This avoids the brittle df.drop(0, inplace=True)
df = df.iloc[1:].reset_index(drop=True)

# Drop unwanted columns if they exist (errors='ignore' avoids exceptions)
df = df.drop(columns=["Active Cases", "Tests"], errors="ignore")

# --- Clean numeric-like string columns robustly ---
numeric_cols = ["Cases", "Deaths", "Recoveries"]
# Remove commas, NBSP, strip, and normalize common placeholder dashes to NaN; then convert to numeric
placeholders = {"–": np.nan, "—": np.nan, "-": np.nan, "": np.nan, "N/A": np.nan, "n/a": np.nan}

for col in numeric_cols:
    if col in df.columns:
        # Ensure we operate on strings safely
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.replace("\u00A0", "", regex=False).str.strip()
        # Convert literal 'nan' strings to actual NaN
        df[col] = df[col].replace({"nan": np.nan})
        # Map common dash/placeholder variants to NaN
        df[col] = df[col].replace(placeholders)
        # Finally convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Parse Date column if present
if "Date" in df.columns:
    df["Date"] = df["Date"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# --- Province cleaning (keeps your approach but slightly more robust) ---
def clean_province_name(name):
    if pd.isna(name):
        return name
    s = str(name).strip()
    # Keep letters, spaces, hyphens and apostrophes; replace other chars with space
    s = re.sub(r"[^A-Za-z\s\-\']", " ", s)
    # Remove the word 'province' if present
    s = re.sub(r"\bprovince\b", " ", s, flags=re.IGNORECASE)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()

if "Province" in df.columns:
    df["Province"] = df["Province"].apply(clean_province_name)

# Apply your corrections mapping
province_corrections = {
    'Jawzjon': 'Jawzjan',
    'Herat': 'Hirat',
    'Sar-E-Pul': 'Sar-E Pol',
    'Jowzjan': 'Jawzjan',
    'Nimruz': 'Nimroz',
    'Nooristan': 'Nuristan',
    'Panjshir': 'Panjsher',
    'Paktia': 'Paktya',
    'Daykundi': 'Dykundi'
}
if "Province" in df.columns:
    df["Province"] = df["Province"].apply(lambda x: province_corrections.get(x, x))

# --- Fill missing numeric values by province median with overall fallback ---
for col in numeric_cols:
    if col not in df.columns:
        continue
    overall_med = df[col].median()
    if "Province" in df.columns:
        # transform returns per-row aligned series of group medians
        group_med = df.groupby("Province")[col].transform("median")
        # fallback to overall median where group median is NaN
        group_med = group_med.fillna(overall_med)
        df[col] = df[col].fillna(group_med)
    else:
        df[col] = df[col].fillna(overall_med)

# --- Fix Recoveries consistency ---
# Ensure 0 <= Recoveries <= Cases - Deaths when Cases and Deaths are present
if set(["Cases", "Deaths", "Recoveries"]).issubset(df.columns):
    # Fill any remaining Recoveries NaNs (safety): use province median fallback already done above
    # Compute max possible recoveries per row
    max_rec = (df["Cases"] - df["Deaths"])
    # cap negative max_rec to 0 (can't have negative possible recoveries)
    max_rec = max_rec.clip(lower=0)
    # For rows where max_rec is not NaN, cap recoveries to max_rec
    mask_has_bound = ~max_rec.isna()
    if mask_has_bound.any():
        # Count how many will be capped
        too_large = (df.loc[mask_has_bound, "Recoveries"] > max_rec.loc[mask_has_bound]).sum()
        df.loc[mask_has_bound, "Recoveries"] = df.loc[mask_has_bound, "Recoveries"].clip(lower=0, upper=max_rec.loc[mask_has_bound])
    else:
        too_large = 0
    # Any negative recoveries (where max_rec was NaN or otherwise) set to 0
    neg_count = (df["Recoveries"] < 0).sum()
    df.loc[df["Recoveries"] < 0, "Recoveries"] = 0
    print(f"Recoveries adjusted: capped_too_large={int(too_large)}, negatives_clipped={int(neg_count)}")
else:
    print("Skipping recoveries consistency (missing Cases/Deaths/Recoveries columns).")

# --- Remove outliers using IQR rule (drop rows outside bounds) ---
# We'll drop rows that are outliers in any of the numeric columns listed (but keep rows where the column is NaN)
outlier_cols = [c for c in numeric_cols if c in df.columns]
if outlier_cols:
    mask = pd.Series(True, index=df.index)
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if pd.isna(IQR) or IQR == 0:
            # If there's no variability, don't exclude rows by this column
            continue
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # Keep rows where value is between bounds OR is NaN (so missing stays)
        mask &= df[col].between(lower, upper) | df[col].isna()
    before = len(df)
    df = df[mask].reset_index(drop=True)
    after = len(df)
    dropped = before - after
    print(f"Outlier removal: dropped {int(dropped)} rows (from {before} to {after})")
else:
    print("No numeric columns found to apply outlier removal.")

# --- Clean column names lightly (just strip spaces) and sort by Date if present ---
df.columns = df.columns.str.strip()
if "Date" in df.columns:
    df = df.sort_values(by="Date").reset_index(drop=True)

# Save cleaned data
df.to_csv(OUTPUT, index=False)
print("Saved cleaned data to", OUTPUT)
