import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def clean_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-Item columns to numeric (remove commas, brackets, spaces)."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    first_col = df.columns[0]
    df.rename(columns={first_col: "Item"}, inplace=True)

    for col in df.columns[1:]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_year_cols(df: pd.DataFrame):
    """Treat all columns except 'Item' as year-like columns."""
    return [c for c in df.columns if c != "Item"]


def avg_revenue_growth(revenue_row: pd.DataFrame, year_cols) -> float:
    """Your revenue growth formula: (last - old) / last, averaged across history."""
    vals = revenue_row[year_cols].values.flatten().astype(float)
    growth = []
    for i in range(1, len(vals)):
        prev_ = vals[i - 1]
        curr_ = vals[i]
        if curr_ != 0:
            g = (curr_ - prev_) / curr_
            if -0.5 < g < 0.5:  # ignore crazy spikes
                growth.append(g)
    if len(growth) == 0:
        return 0.05
    return float(np.mean(growth))


def ratio_to_revenue(row_vals: np.ndarray, rev_vals: np.ndarray) -> float:
    """Average (row / revenue) on overlapping years."""
    mask = (~np.isnan(row_vals)) & (~np.isnan(rev_vals)) & (rev_vals != 0)
    if not mask.any():
        return 0.0
    ratios = row_vals[mask] / rev_vals[mask]
    ratios = ratios[(ratios > -5) & (ratios < 5)]
    if len(ratios) == 0:
        return 0.0
    return float(np.mean(ratios))


def find_row_indices(df: pd.DataFrame, keywords):
    """Return list of index positions whose 'Item' contains any keyword."""
    if df.empty:
        return []
    s = df["Item"].astype(str).str.lower()
    mask = False
    for kw in keywords:
        mask = mask | s.str.contains(kw, na=False)
    return list(df[mask].index)


def find_single_row(df: pd.DataFrame, keywords):
    idx_list = find_row_indices(df, keywords)
    if not idx_list:
        return None, None
    idx = idx_list[0]
    return idx, df.iloc[idx]


def fetch_rbz_fx_yearly() -> dict | None:
    """
    Try to fetch RBZ exchange-rate page and compute average yearly USD rate.
    Returns dict: { '2023': rate, '2024': rate, ... } or None on failure.
    """
    url = "https://www.rbz.co.zw/index.php/research/markets/exchange-rates"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception:
        return None

    try:
        tables = pd.read_html(r.text)
    except Exception:
        return None

    df = None
    for t in tables:
        if any("date" in str(c).lower() for c in t.columns):
            df = t
            break
    if df is None:
        return None

    df.columns = [str(c) for c in df.columns]
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        return None

    usd_col = None
    for c in df.columns:
        if "usd" in c.lower() or "us$" in c.lower():
            usd_col = c
            break
    if usd_col is None and len(df.columns) >= 2:
        usd_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[usd_col] = pd.to_numeric(
        df[usd_col].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    df = df.dropna(subset=[date_col, usd_col])
    if df.empty:
        return None

    df["Year"] = df[date_col].dt.year
    yearly = df.groupby("Year")[usd_col].mean().round(4)
    return {str(int(y)): float(rate) for y, rate in yearly.items()}


def convert_df_yearwise(df: pd.DataFrame, year_rates: dict) -> pd.DataFrame:
    """Divide each year column by its matching FX rate, to convert ZWL → USD."""
    df2 = df.copy()
    for col in df2.columns:
        if col == "Item":
            continue
        key = str(col)
        if key in year_rates and year_rates[key] != 0:
            df2[col] = df2[col] / year_rates[key]
    return df2


def option_labels_from_items(items):
    """Build labels like '3: Inventories' for selectboxes."""
    return [f"{i+1}: {name}" for i, name in enumerate(items)]


def indices_from_labels(labels):
    """Parse ['3: Inventories', ...] → [2, ...] (0-based indices)."""
    idx = []
    for s in labels:
        try:
            i = int(str(s).split(":", 1)[0]) - 1
            idx.append(i)
        except Exception:
            continue
    return idx


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Forecast + DCF (IS + BS + CF + Mapping + WC from BS)",
    layout="wide"
)
st.title("📊 Forecast + DCF Valuation (IS + BS + CF + Working Capital from Balance Sheet)")

uploaded_file = st.file_uploader(
    "Upload Excel with Income Statement, Balance Sheet and Cash Flow sheets",
    type=["xlsx"]
)

if not uploaded_file:
    st.stop()

xls = pd.ExcelFile(uploaded_file)
st.write("Detected sheets:", xls.sheet_names)

is_sheet = st.selectbox("Income Statement sheet", xls.sheet_names, index=0)
bs_sheet = st.selectbox("Balance Sheet sheet", xls.sheet_names, index=min(1, len(xls.sheet_names) - 1))
cf_sheet = st.selectbox("Cash Flow sheet", xls.sheet_names, index=min(2, len(xls.sheet_names) - 1))

# ---------------------------------------------------------
# LOAD & CLEAN RAW SHEETS
# ---------------------------------------------------------
is_df = clean_numeric_cols(xls.parse(is_sheet))
bs_df = clean_numeric_cols(xls.parse(bs_sheet))
cf_df = clean_numeric_cols(xls.parse(cf_sheet))

year_cols_is = get_year_cols(is_df)
year_cols_bs = get_year_cols(bs_df)
year_cols_cf = get_year_cols(cf_df)

# ---------------------------------------------------------
# FX SECTION
# ---------------------------------------------------------
st.markdown("### 💱 Currency & Exchange Rates")

currency = st.selectbox(
    "Currency of uploaded statements:",
    ["USD (already converted)", "ZWL/ZWG (need RBZ FX conversion)"],
    index=0
)

year_rates = {}

if currency.startswith("USD"):
    st.success("✅ Data assumed to be in USD. No FX conversion applied.")
else:
    st.warning("Data is in ZWL/ZWG. Convert to USD using FX rates.")
    fx_mode = st.radio(
        "How do you want to get ZWL → USD exchange rates?",
        ["Fetch automatically from RBZ website", "Enter per-year rates manually"],
    )

    if fx_mode == "Fetch automatically from RBZ website":
        if st.button("🌐 Fetch RBZ exchange rates now"):
            rbz_rates = fetch_rbz_fx_yearly()
            if rbz_rates is None or len(rbz_rates) == 0:
                st.error("❌ Could not fetch or parse RBZ exchange rates. Please use manual mode.")
            else:
                st.success(f"RBZ yearly FX rates found: {rbz_rates}")
                for y in year_cols_is:
                    default_rate = rbz_rates.get(str(y), 15000.0)
                    rate = st.number_input(
                        f"FX rate for {y} (ZWL per USD)",
                        min_value=0.0001,
                        value=float(default_rate),
                        step=1.0,
                        format="%.4f"
                    )
                    year_rates[str(y)] = rate
        if not year_rates:
            st.info("Click the button above to fetch rates, then confirm/override them.")
            st.stop()
    else:
        for y in year_cols_is:
            rate = st.number_input(
                f"FX rate for {y} (ZWL per USD)",
                min_value=0.0001,
                value=15000.0,
                step=1.0,
                format="%.4f"
            )
            year_rates[str(y)] = rate

    is_df = convert_df_yearwise(is_df, year_rates)
    bs_df = convert_df_yearwise(bs_df, year_rates)
    cf_df = convert_df_yearwise(cf_df, year_rates)
    st.success(f"✅ Financials converted to USD using: {year_rates}")

# ---------------------------------------------------------
# SHOW CLEANED STATEMENTS
# ---------------------------------------------------------
st.subheader("Income Statement (cleaned, in USD)")
st.dataframe(is_df, use_container_width=True)

st.subheader("Balance Sheet (cleaned, in USD)")
st.dataframe(bs_df, use_container_width=True)

st.subheader("Cash Flow Statement (cleaned, in USD)")
st.dataframe(cf_df, use_container_width=True)

# Re-detect year columns (as strings)
year_cols_is = get_year_cols(is_df)
year_cols_bs = get_year_cols(bs_df)
year_cols_cf = get_year_cols(cf_df)

if len(year_cols_is) < 2:
    st.error("❌ Need at least 2 historical year columns in Income Statement.")
    st.stop()

# Prepare year ints/labels
last_hist_label = year_cols_is[-1]           # string label e.g. "2025"
last_hist_year = int(str(last_hist_label))   # int 2025
forecast_years_int = [last_hist_year + i for i in range(1, 6)]
forecast_cols = [str(y) for y in forecast_years_int]

# ---------------------------------------------------------
# BALANCE SHEET MAPPING (multi-select)
# ---------------------------------------------------------
st.markdown("### 🟩 Balance Sheet Mapping (multi-select allowed)")

bs_items = list(bs_df["Item"].astype(str))
bs_labels = option_labels_from_items(bs_items)

sel_debt_labels = st.multiselect(
    "Select ALL rows that form Total Debt / Borrowings:",
    bs_labels
)
sel_cash_labels = st.multiselect(
    "Select ALL rows that form Cash & Cash Equivalents:",
    bs_labels
)
sel_ca_labels = st.multiselect(
    "Select ALL rows that are Current Assets (for Working Capital):",
    bs_labels
)
sel_cl_labels = st.multiselect(
    "Select ALL rows that are Current Liabilities (for Working Capital):",
    bs_labels
)

debt_idx_list = indices_from_labels(sel_debt_labels)
cash_idx_list = indices_from_labels(sel_cash_labels)
ca_idx_list = indices_from_labels(sel_ca_labels)
cl_idx_list = indices_from_labels(sel_cl_labels)

# ---------------------------------------------------------
# CASH FLOW MAPPING (multi-select)
# ---------------------------------------------------------
st.markdown("### 📄 Cash Flow Mapping (multi-select allowed)")

cf_items = list(cf_df["Item"].astype(str))
cf_labels = option_labels_from_items(cf_items)

sel_dep_cf = st.multiselect(
    "Select Depreciation & Amortisation rows (from Cash Flow):",
    cf_labels
)
sel_capex_cf = st.multiselect(
    "Select ALL Capex rows (purchase of PPE / fixed assets):",
    cf_labels
)
sel_int_cf = st.multiselect(
    "Select Interest paid rows (if using CF for interest):",
    cf_labels
)

dep_cf_idx_list = indices_from_labels(sel_dep_cf)
capex_cf_idx_list = indices_from_labels(sel_capex_cf)
int_cf_idx_list = indices_from_labels(sel_int_cf)

# ---------------------------------------------------------
# INCOME STATEMENT FORECASTING
# ---------------------------------------------------------
# find main rows
rev_idx, _ = find_single_row(is_df, ["revenue"])
cos_idx, _ = find_single_row(is_df, ["cost of sales"])
gp_idx, _ = find_single_row(is_df, ["gross profit"])
ebitda_idx, _ = find_single_row(is_df, ["ebitda"])
op_idx, _ = find_single_row(is_df, ["operating profit"])
pbt_idx, _ = find_single_row(is_df, ["profit before tax"])
np_idx, _ = find_single_row(is_df, ["profit for the year", "profit for the period"])

if rev_idx is None or cos_idx is None:
    st.error("❌ Could not find both 'Revenue' and 'Cost of sales' rows.")
    st.stop()

revenue_row = is_df.iloc[[rev_idx]]

# revenue growth
avg_g = avg_revenue_growth(revenue_row, year_cols_is)
st.info(f"📌 Average revenue growth (your formula (last-old)/last): **{avg_g:.2%}**")

# build forecast IS
forecast_is = is_df.copy()
for col in forecast_cols:
    if col not in forecast_is.columns:
        forecast_is[col] = np.nan

# revenue forecast
rev_hist_vals = revenue_row[year_cols_is].values.flatten().astype(float)
rev_forecast = {}
last_rev_val = rev_hist_vals[-1]
current_rev = last_rev_val
for y in forecast_years_int:
    current_rev = current_rev * (1 + avg_g)
    rev_forecast[y] = current_rev
    col = str(y)
    forecast_is.iat[rev_idx, forecast_is.columns.get_loc(col)] = current_rev

# cost of sales via GP margin
gp_idx_for_margin = gp_idx
if gp_idx_for_margin is None:
    st.warning("No 'Gross profit' row found – GP margin method for Cost of sales disabled.")
else:
    gp_hist_vals = forecast_is.iloc[gp_idx_for_margin][year_cols_is].values.astype(float)
    mask = rev_hist_vals != 0
    gp_margins = np.zeros_like(rev_hist_vals, dtype=float)
    gp_margins[mask] = gp_hist_vals[mask] / rev_hist_vals[mask]
    gp_margins = gp_margins[(gp_margins > -5) & (gp_margins < 5)]
    if len(gp_margins) == 0:
        avg_gp_margin = 0.3
    else:
        avg_gp_margin = float(np.mean(gp_margins))

    last_cos_hist = float(forecast_is.iloc[cos_idx][last_hist_label])
    cos_sign = -1 if last_cos_hist < 0 else 1

    for y in forecast_years_int:
        rev_y = rev_forecast[y]
        cos_y = cos_sign * rev_y * (1 - avg_gp_margin)
        forecast_is.iat[cos_idx, forecast_is.columns.get_loc(str(y))] = cos_y

# forecast other non-total, non-CoS rows as % of revenue
total_keywords = ["gross profit", "ebitda", "operating profit",
                  "profit before tax", "profit for the year", "profit for the period"]

for idx in range(len(forecast_is)):
    if idx in [rev_idx, cos_idx, gp_idx, ebitda_idx, op_idx, pbt_idx, np_idx]:
        continue
    item = str(forecast_is.at[idx, "Item"]).lower()
    if any(k in item for k in total_keywords):
        continue

    row_hist_vals = forecast_is.iloc[idx][year_cols_is].values.astype(float)
    ratio = ratio_to_revenue(row_hist_vals, rev_hist_vals)
    for y in forecast_years_int:
        forecast_is.iat[idx, forecast_is.columns.get_loc(str(y))] = rev_forecast[y] * ratio

# recompute totals for forecast years
def sum_rows(df, start_idx, end_idx, col):
    """Sum from start_idx to end_idx-1 inclusive."""
    if start_idx is None or end_idx is None:
        return df.iloc[start_idx][col] if start_idx is not None else np.nan
    if end_idx <= start_idx:
        return df.iloc[start_idx][col]
    return df.loc[start_idx:end_idx-1, col].sum(skipna=True)

for col in forecast_cols:
    # Gross profit = Revenue + Cost of sales (since CoS usually negative)
    if gp_idx is not None:
        rev_val = forecast_is.iloc[rev_idx][col]
        cos_val = forecast_is.iloc[cos_idx][col]
        forecast_is.iat[gp_idx, forecast_is.columns.get_loc(col)] = rev_val + cos_val

    # EBITDA = sum from GP row to row before EBITDA
    if ebitda_idx is not None and gp_idx is not None:
        ebitda_val = sum_rows(forecast_is, gp_idx, ebitda_idx, col)
        forecast_is.iat[ebitda_idx, forecast_is.columns.get_loc(col)] = ebitda_val

    # Operating profit
    if op_idx is not None and ebitda_idx is not None:
        op_val = sum_rows(forecast_is, ebitda_idx, op_idx, col)
        forecast_is.iat[op_idx, forecast_is.columns.get_loc(col)] = op_val

    # Profit before tax
    if pbt_idx is not None and op_idx is not None:
        pbt_val = sum_rows(forecast_is, op_idx, pbt_idx, col)
        forecast_is.iat[pbt_idx, forecast_is.columns.get_loc(col)] = pbt_val

    # Profit for year
    if np_idx is not None and pbt_idx is not None:
        np_val = sum_rows(forecast_is, pbt_idx, np_idx, col)
        forecast_is.iat[np_idx, forecast_is.columns.get_loc(col)] = np_val

st.subheader("📘 Forecasted Income Statement (5 years, USD)")
st.dataframe(
    forecast_is.style.format(
        {c: "{:,.0f}".format for c in forecast_is.select_dtypes(include=[np.number]).columns},
        na_rep="",
    ),
    use_container_width=True,
)

# Extract EBITDA row for forecast years
ebitda_forecast_vals = np.array(
    [forecast_is.iloc[ebitda_idx][str(y)] for y in forecast_years_int],
    dtype=float
) if ebitda_idx is not None else np.zeros(len(forecast_years_int))

# Depreciation from IS if present
dep_hist_from_is_idx, _ = find_single_row(forecast_is, ["depreciation"])
if dep_hist_from_is_idx is not None:
    dep_forecast_vals = np.array(
        [forecast_is.iloc[dep_hist_from_is_idx][str(y)] for y in forecast_years_int],
        dtype=float
    )
else:
    # fallback to CF-based ratio (rarely used now)
    if dep_cf_idx_list:
        common = [c for c in year_cols_cf if c in year_cols_is]
        dep_ratio = ratio_to_revenue(
            cf_df.loc[dep_cf_idx_list, common].sum(axis=0).values.astype(float),
            revenue_row[common].values.flatten().astype(float)
        )
    else:
        dep_ratio = 0.0
    dep_forecast_vals = np.array(
        [rev_forecast[y] * dep_ratio for y in forecast_years_int],
        dtype=float
    )

# ---------------------------------------------------------
# CAPITAL STRUCTURE FROM BS: Total Debt, Cash, CA, CL
# ---------------------------------------------------------
common_hist_bs = [c for c in year_cols_bs if c in year_cols_is]
bs_year_used_label = common_hist_bs[-1] if common_hist_bs else year_cols_bs[-1]

total_debt = 0.0
if debt_idx_list:
    total_debt = float(bs_df.loc[debt_idx_list, bs_year_used_label].sum(skipna=True))

cash_bal = 0.0
if cash_idx_list:
    cash_bal = float(bs_df.loc[cash_idx_list, bs_year_used_label].sum(skipna=True))

# equity: try some standard keywords
eq_idx_list = find_row_indices(bs_df, ["shareholders' equity", "total equity", "equity attributable"])
total_equity = float(bs_df.loc[eq_idx_list, bs_year_used_label].sum(skipna=True)) if eq_idx_list else 0.0

net_debt = total_debt - cash_bal
de_ratio = (total_debt / total_equity) if total_equity != 0 else 0.0
# ---------------------------------------------------------
# 🟦 WORKING CAPITAL MODULE (HISTORICAL → WC% → FORECAST → ΔWC)
# ---------------------------------------------------------

st.subheader("📘 Working Capital Calculation (Historical & Forecast)")

# Ensure lists exist
delta_wc_forecast_vals = np.zeros(len(forecast_years_int))

if ca_idx_list and cl_idx_list:

    # -------- 1️⃣ HISTORICAL WC (CA - CL)
    st.markdown("### **Historical Working Capital (CA - CL)**")

    ca_hist = bs_df.loc[ca_idx_list, year_cols_bs].sum(axis=0)
    cl_hist = bs_df.loc[cl_idx_list, year_cols_bs].sum(axis=0)
    wc_hist = ca_hist - cl_hist

    df_wc_hist = pd.DataFrame({
        "Year": year_cols_bs,
        "Current Assets": ca_hist.values,
        "Current Liabilities": cl_hist.values,
        "Working Capital (CA-CL)": wc_hist.values,
    })

    st.dataframe(
        df_wc_hist.style.format({
            "Current Assets": "{:,.0f}",
            "Current Liabilities": "{:,.0f}",
            "Working Capital (CA-CL)": "{:,.0f}",
        }),
        use_container_width=True
    )
    # --------------- 2️⃣ WC% OF SALES ----------------
    st.markdown("### **Historical Working Capital as % of Sales**")

    # Only use years available in both IS and BS
    common_hist = [c for c in year_cols_is if c in wc_hist.index]

    wc_vals_hist = wc_hist[common_hist].astype(float).values
    rev_vals_hist = revenue_row[common_hist].values.flatten().astype(float)

    # Raw WC% of sales for each common year
    wc_percent_hist = wc_vals_hist / rev_vals_hist

    df_wc_pct = pd.DataFrame({
        "Year": common_hist,
        "Working Capital": wc_vals_hist,
        "Revenue": rev_vals_hist,
        "WC % of Sales": wc_percent_hist,
    })

    st.dataframe(
        df_wc_pct.style.format({
            "Working Capital": "{:,.0f}".format,
            "Revenue": "{:,.0f}".format,
            "WC % of Sales": "{:.2%}".format,
        }),
        use_container_width=True
    )

    # --------------- 3️⃣ AVERAGE WC% WITH OUTLIER HANDLING ----------------
    # Filter out totally crazy ratios first (e.g. >500% or <-500%)
    wc_percent_array = wc_percent_hist.copy()
    mask_valid = (wc_percent_array > -5) & (wc_percent_array < 5)
    wc_percent_clean = wc_percent_array[mask_valid]

    if len(wc_percent_clean) == 0:
        wc_percent_avg = 0.0
        st.warning("No valid WC% of sales ratios found – using 0% by default.")
    else:
        # Measure how wide the range is between min and max ratio
        ratio_spread = float(wc_percent_clean.max() - wc_percent_clean.min())

        # Threshold for “similar range” – here 5 percentage points
        spread_threshold = 0.05  # 0.05 = 5%

        if ratio_spread > spread_threshold and len(wc_percent_clean) >= 2:
            # 🔹 Ratios are very far apart → treat as outlier situation
            # Use the *most recent* WC% (last available year) instead of the average.
            last_year = common_hist[-1]
            last_wc = float(wc_hist[last_year])
            last_rev = float(revenue_row[last_year].values[0])
            wc_percent_avg = last_wc / last_rev

            st.warning(
                f"WC% of sales ratios differ a lot (spread ≈ {ratio_spread:.2%}). "
                f"Using the **most recent WC% ({wc_percent_avg:.2%})** for forecasting "
                f"instead of the average."
            )
        else:
            # 🔹 Ratios are in a similar band → use average of all valid ratios
            wc_percent_avg = float(np.mean(wc_percent_clean))
            st.success(
                f"WC% of sales ratios are in a similar range (spread ≈ {ratio_spread:.2%}). "
                f"Using the **average WC% = {wc_percent_avg:.2%}** for forecasting."
            )


    # -------- 3️⃣ FORECAST WC
    st.markdown("### **Forecast Working Capital**")

    wc_forecast_vals = np.array(
        [rev_forecast[y] * wc_percent_avg for y in forecast_years_int],
        dtype=float
    )

    df_wc_forecast = pd.DataFrame({
        "Year": forecast_years_int,
        "Forecast Revenue": [rev_forecast[y] for y in forecast_years_int],
        "Forecast WC": wc_forecast_vals,
    })

    st.dataframe(
        df_wc_forecast.style.format({
            "Forecast Revenue": "{:,.0f}",
            "Forecast WC": "{:,.0f}",
        }),
        use_container_width=True
    )

    # -------- 4️⃣ ΔWC = OLD – NEW
    st.markdown("### **Change in Working Capital (ΔWC = Old – New)**")

    last_wc_hist_value = float(wc_hist[common_hist[-1]])  # last historical WC

    prev_wc = last_wc_hist_value
    delta_list = []

    for wc_new in wc_forecast_vals:
        delta_list.append(prev_wc - wc_new)  # Old – New
        prev_wc = wc_new

    delta_wc_forecast_vals = np.array(delta_list, dtype=float)

    df_delta_wc = pd.DataFrame({
        "Year": forecast_years_int,
        "Forecast WC": wc_forecast_vals,
        "ΔWC (Old – New)": delta_wc_forecast_vals,
    })

    st.dataframe(
        df_delta_wc.style.format({
            "Forecast WC": "{:,.0f}",
            "ΔWC (Old – New)": "{:,.0f}",
        }),
        use_container_width=True
    )

else:
    st.warning("⚠️ Please select Current Assets and Current Liabilities rows first.")


st.subheader("Capital Structure & Working Capital (from Balance Sheet)")
c_cap1, c_cap2, c_cap3, c_cap4 = st.columns(4)
with c_cap1:
    st.metric(f"Total Debt ({bs_year_used_label})", f"{total_debt:,.0f}")
with c_cap2:
    st.metric(f"Cash & Equivalents ({bs_year_used_label})", f"{cash_bal:,.0f}")
with c_cap3:
    st.metric(f"Net Debt", f"{net_debt:,.0f}")
with c_cap4:
    st.metric("D/E Ratio", f"{de_ratio:.2f}x")

# ---------------------------------------------------------
# CAPEX: average historical PPE / capex from CF
# ---------------------------------------------------------
avg_capex = 0.0
if capex_cf_idx_list:
    common_capex = [c for c in year_cols_cf if c in year_cols_is]
    if common_capex:
        capex_hist_vals = cf_df.loc[capex_cf_idx_list, common_capex].sum(axis=0).values.astype(float)
        if len(capex_hist_vals) > 0:
            avg_capex = float(np.nanmean(capex_hist_vals))

capex_forecast_vals = np.full(len(forecast_years_int), avg_capex, dtype=float)

# ---------------------------------------------------------
# COST OF DEBT (Interest / Debt)
# ---------------------------------------------------------
int_is_idx_list = find_row_indices(is_df, ["net finance costs", "finance costs", "interest expense", "interest paid"])
if int_is_idx_list:
    interest_last = float(is_df.loc[int_is_idx_list, last_hist_label].sum(skipna=True))
else:
    if int_cf_idx_list:
        interest_last = float(cf_df.loc[int_cf_idx_list, bs_year_used_label].sum(skipna=True))
    else:
        interest_last = 0.0

if total_debt != 0:
    cost_of_debt = abs(interest_last) / abs(total_debt)
else:
    cost_of_debt = 0.0

# ---------------------------------------------------------
# DCF PARAMETERS (USER INPUT)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("💰 DCF Parameters")

col1, col2 = st.columns(2)
with col1:
    rf_pct = st.number_input("Risk-free rate (%)", value=11.61, step=0.1)
    mrp_pct = st.number_input("Market risk premium (%)", value=13.82, step=0.1)
    tax_pct = st.number_input("Tax rate (%)", value=25.0, step=0.5)
with col2:
    unlevered_beta = st.number_input("Unlevered beta (asset beta)", value=1.00, step=0.05)
    terminal_g_pct = st.number_input("Terminal growth rate (%)", value=5.0, step=0.1)

rf = rf_pct / 100.0
mrp = mrp_pct / 100.0
tax = tax_pct / 100.0
g = terminal_g_pct / 100.0
rd = cost_of_debt

# Levered beta (Hamada)
beta_levered = unlevered_beta * (1 + (1 - tax) * de_ratio)

# Capital structure weights
if de_ratio <= 0:
    w_d = 0.0
    w_e = 1.0
else:
    w_d = de_ratio / (1.0 + de_ratio)
    w_e = 1.0 / (1.0 + de_ratio)

re = rf + beta_levered * mrp
wacc = w_e * re + w_d * rd * (1 - tax)

st.markdown(
    f"""
**Auto Cost of Debt (Interest / Debt):** {rd*100:.2f}%  
**Levered Beta (βₗ):** {beta_levered:.2f}  
**Cost of Equity (Re):** {re*100:.2f}%  
**WACC:** {wacc*100:.2f}%  
"""
)
# ---------------------------------------------------------
# SAVE DCF PARAMETERS IN SESSION STATE
# ---------------------------------------------------------
st.session_state["rf"] = rf       # risk-free rate (decimal)
st.session_state["erp"] = mrp     # equity risk premium (decimal)
st.session_state["tax_rate"] = tax
st.session_state["de_ratio"] = de_ratio
st.session_state["unlevered_beta"] = unlevered_beta
st.session_state["levered_beta"] = beta_levered
st.session_state["wacc"] = wacc


# ---------------------------------------------------------
# DATE-BASED DISCOUNTING: valuation date + FS date
# ---------------------------------------------------------
st.markdown("### 📅 Valuation Timing & Mid-point")

today_default = date.today()
valuation_date = st.date_input(
    "Valuation date (today / deal date)",
    value=today_default,
)

first_forecast_fs_date = st.date_input(
    "Financial statement year-end date for forecasts (first forecast year)",
    value=date(last_hist_year + 1, 12, 31),
)

use_midyear = st.checkbox("Use mid-year (0.5 year earlier) convention?", value=False)

gap_days = (first_forecast_fs_date - valuation_date).days
gap_years = gap_days / 365.25

n0 = max(gap_years, 0.0)
if use_midyear:
    n0 = max(n0 - 0.5, 0.0)

discount_periods_n = np.array([n0 + i for i in range(len(forecast_years_int))], dtype=float)

midpoint_df0 = (1 / (1 + wacc) ** n0) if wacc > 0 else 1.0

midpoint_table = pd.DataFrame(
    {
        "Valuation date": [valuation_date],
        "FS date (first forecast year)": [first_forecast_fs_date],
        "Gap (days)": [gap_days],
        "Discount period n₀ (years)": [n0],
        "Mid-point DF₀ = 1/(1+WACC)ⁿ⁰": [midpoint_df0],
    }
)
st.dataframe(midpoint_table, use_container_width=True)

# ---------------------------------------------------------
# FCFF / UFCF
# ---------------------------------------------------------
ebitda_after_tax = ebitda_forecast_vals * (1 - tax)
dep_tax_vals = dep_forecast_vals * tax

# UFCF = EBITDA(1-T) + Dep×T + ΔWC + Capex
fcff_vals = ebitda_after_tax + dep_tax_vals + delta_wc_forecast_vals + capex_forecast_vals

# Discount factors using date-based n
discount_factors = np.array([(1 / (1 + wacc) ** n) for n in discount_periods_n])
pv_fcff = fcff_vals * discount_factors

# ---------------------------------------------------------
# TERMINAL VALUE  (FIXED — now uses last DCF discount factor)
# ---------------------------------------------------------
if wacc <= g:
    terminal_value = np.nan
    pv_terminal = np.nan
else:
    # Standard perpetual growth formula
    terminal_value = fcff_vals[-1] * (1 + g) / (wacc - g)

    # ✅ FIX: terminal DF must equal LAST discount factor in DCF table
    discount_factor_terminal = float(discount_factors[-1])

    # Present value of Terminal Value
    pv_terminal = terminal_value * discount_factor_terminal



enterprise_value = np.nansum(pv_fcff) + (0 if np.isnan(pv_terminal) else pv_terminal)
equity_value = enterprise_value - net_debt

# ---------------------------------------------------------
# DCF TABLE (UFCF style)
# ---------------------------------------------------------
st.subheader("📉 DCF Cashflows (UFCF) — Date-based Discounting")

df_dcf = pd.DataFrame(
    {
        "Year": [str(y) for y in forecast_years_int],  # avoid thousand-separator
        "Discount period n (years)": discount_periods_n,
        "EBITDA × (1−T)": ebitda_after_tax,
        "Depreciation × Tax": dep_tax_vals,
        "Δ Working capital": delta_wc_forecast_vals,
        "Capex": capex_forecast_vals,
        "UFCF": fcff_vals,
        "Discount factor": discount_factors,
        "PV of UFCF": pv_fcff,
    }
)

num_cols_dcf = df_dcf.select_dtypes(include=[np.number]).columns
fmt_dict = {c: "{:,.0f}".format for c in num_cols_dcf if c not in ["Discount period n (years)", "Discount factor"]}
fmt_dict["Discount period n (years)"] = "{:.3f}".format
fmt_dict["Discount factor"] = "{:.3f}".format

styled_dcf = df_dcf.style.format(fmt_dict, na_rep="")
st.dataframe(styled_dcf, use_container_width=True)

# Terminal summary
st.write("**Terminal Value and Present Value:**")

df_term = pd.DataFrame(
    {
        "Terminal Value": [terminal_value],
        "Discount factor (last year)": [discount_factors[-1]],
        "PV of Terminal Value": [pv_terminal],
    }
)

# Apply proper formatting
fmt_term = {}
for c in df_term.columns:
    if c == "Discount factor (last year)":
        fmt_term[c] = "{:.3f}".format   # show decimals
    else:
        fmt_term[c] = "{:,.0f}".format  # round whole values

st.dataframe(
    df_term.style.format(fmt_term, na_rep=""),
    use_container_width=True,
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
st.subheader("📌 Valuation Summary")

c_sum1, c_sum2 = st.columns(2)
with c_sum1:
    st.metric("Enterprise Value (EV)", f"{enterprise_value:,.0f}")
    st.metric("Net Debt", f"{net_debt:,.0f}")
with c_sum2:
    st.metric("Equity Value", f"{equity_value:,.0f}")
    st.metric("WACC", f"{wacc*100:.2f}%")
    st.metric("Terminal Growth Rate", f"{g*100:.2f}%")
