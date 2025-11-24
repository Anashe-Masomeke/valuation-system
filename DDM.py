import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dividend Discount Model (DDM)", layout="wide")
st.title("📈 Dividend Discount Model (DDM)")

st.markdown("""
This module values equity using the Gordon Growth Dividend Discount Model:

### **P₀ = D₁ / (Re − g)**  
Where:  
- **D₁ = Dividend next year**  
- **Re = Cost of Equity**  
- **g = perpetual growth rate**
""")

# STEP 1 — SELECT YEAR RANGE + ENTER DIVIDENDS
# ---------------------------------------------------------
st.header("📘 Step 1 — Select Year Range & Enter Dividends")

st.info("Select start and end years. The system will auto-generate all years in between.")

colY1, colY2 = st.columns(2)
with colY1:
    start_year_input = st.number_input("Start Year", value=2021, step=1)
with colY2:
    end_year_input = st.number_input("End Year", value=2025, step=1)

# Validate
if start_year_input > end_year_input:
    st.error("❌ Start year cannot be greater than end year.")
    st.stop()

# Generate years
years = list(range(int(start_year_input), int(end_year_input) + 1))

st.write("### Enter Dividends for Each Year")

dividends = []
for y in years:
    div = st.number_input(
        f"Dividend for {y}",
        min_value=0.0,
        value=0.01,
        step=0.00001,
        format="%.5f"
    )
    dividends.append(div)

# Display table
df_disp = pd.DataFrame({"Year": years, "Dividend": dividends})
st.dataframe(df_disp, use_container_width=True)

# ---------------------------------------------------------
# STEP 2 — SELECT YEAR RANGE FOR GROWTH CALCULATION
# ---------------------------------------------------------
st.header("📘 Step 2 — Select Years to Use for Growth Calculation")

colG1, colG2 = st.columns(2)
with colG1:
    start_year = st.selectbox("Select growth start year:", years, index=0)
with colG2:
    end_year = st.selectbox("Select growth end year:", years, index=len(years)-1)

if start_year > end_year:
    st.error("❌ Growth start year cannot be greater than end year.")
    st.stop()

# Extract dividends
D_start = dividends[years.index(start_year)]
D_end   = dividends[years.index(end_year)]

# ---------------------------------------------------------
# STEP 3 — GROWTH CALCULATION
# ---------------------------------------------------------
st.header("📘 Step 3 — Dividend Growth (g)")

if end_year == start_year:
    g = 0.0
elif D_start > 0:
    g = (D_end / D_start) ** (1 / (end_year - start_year)) - 1
else:
    g = 0.02  # fallback

st.success(f"Calculated growth rate (g): **{g:.2%}**")

# D1 — Next year's dividend
D1 = D_end * (1 + g)
st.metric("Next year's dividend (D₁)", f"{D1:,.5f}")

# ---------------------------------------------------------
# COST OF EQUITY SECTION
# ---------------------------------------------------------
st.header("📘 Step 4 — Cost of Equity (Re)")

# LOAD DEFAULTS FROM SESSION / DCF PAGE
default_rf = st.session_state.get("rf", 0.0861)
default_erp = st.session_state.get("erp", 0.13825)
default_tax = st.session_state.get("tax_rate", 0.25)
default_unlevered_beta = st.session_state.get("unlevered_beta", 0.45)
default_de = st.session_state.get("de_ratio", 0.54)

use_custom_params = st.checkbox("Manually override ERP, Beta, RF, D/E, Tax?", value=False)

if use_custom_params:

    colA, colB = st.columns(2)

    with colA:
        unlevered_beta = st.number_input(
            "Unlevered Beta",
            value=float(default_unlevered_beta),
            step=0.00001,
            format="%.5f"
        )

        debt_equity = st.number_input(
            "Debt/Equity ratio (D/E)",
            value=float(default_de),
            step=0.00001,
            format="%.5f"
        )

    with colB:
        tax_rate = st.number_input(
            "Tax Rate (%)",
            value=float(default_tax * 100),
            step=0.00001,
            format="%.5f"
        ) / 100

        rf = st.number_input(
            "Risk Free Rate (%)",
            value=float(default_rf * 100),
            step=0.00001,
            format="%.15f"
        ) / 100

        erp = st.number_input(
            "Equity Risk Premium (%)",
            value=float(default_erp * 100),
            step=0.00001,
            format="%.15f"
        ) / 100
else:
    unlevered_beta = float(default_unlevered_beta)
    debt_equity = float(default_de)
    tax_rate = float(default_tax)
    rf = float(default_rf)
    erp = float(default_erp)

# Levered Beta
levered_beta = unlevered_beta * (1 + (1 - tax_rate) * debt_equity)
st.metric("Levered Beta", f"{levered_beta:.4f}")

# Cost of Equity
Re = rf + levered_beta * erp
st.metric("Cost of Equity (Re)", f"{Re*100:.2f}%")

# ---------------------------------------------------------
# STEP 5 — EQUITY VALUE PER SHARE
# ---------------------------------------------------------
st.header("📘 Step 5 — Equity Value Per Share")

if Re <= g:
    st.error("❌ Cost of equity must be greater than the growth rate (Re > g required).")
    P0 = np.nan
else:
    P0 = D1 / (Re - g)
    st.success(f"Equity Value per Share = **{P0:,.5f} USD**")

# ---------------------------------------------------------
# STEP 6 — TOTAL EQUITY VALUE
# ---------------------------------------------------------
st.header("📘 Step 6 — Total Equity Valuation")

num_shares = st.number_input("Number of Shares", value=0, step=1000, format="%.0f")

if not np.isnan(P0):
    equity_val = P0 * num_shares
    st.metric("Total Equity Valuation", f"{equity_val:,.2f} ")
