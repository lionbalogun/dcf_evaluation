# streamlit_app.py
"""
Streamlit DCF tool — ready for Streamlit Cloud
Save as streamlit_app.py and deploy the repository to Streamlit Cloud.
"""
import re
import sys
import math
import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="Pride Advisory — DCF Tool", layout="wide")

# -----------------------
# Parsing helpers (same logic as CLI)
# -----------------------
UNIT_MULTIPLIERS = {
    "": 1.0,
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tr": 1e12, "tn": 1e12, "trillion": 1e12
}

def parse_money(s):
    if s is None:
        return None, ""
    s = str(s).strip().lower()
    if s == "":
        return None, ""
    s = s.replace("₦", "").replace("ngn", "").replace("$", "")
    s = s.replace(",", "")
    m = re.match(r"^(-?\d+(\.\d+)?)(\s*([a-zA-Z]+))?$", s)
    if m:
        num = float(m.group(1))
        unit_token = (m.group(4) or "").strip().lower()
        for token, mult in UNIT_MULTIPLIERS.items():
            if token != "" and (unit_token == token or unit_token.startswith(token)):
                return num * mult, token
        return num, ""
    for word, mult in UNIT_MULTIPLIERS.items():
        if word != "" and word in s:
            cleaned = re.sub(r"[^\d\.\-]", "", s)
            if cleaned == "":
                continue
            try:
                num = float(cleaned)
                return num * mult, word
            except:
                pass
    cleaned = re.sub(r"[^\d\.\-]", "", s)
    if cleaned:
        try:
            return float(cleaned), ""
        except:
            pass
    raise ValueError(f"Could not parse money value: '{s}'")

def parse_percent_or_decimal(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    if "%" in s:
        cleaned = s.replace("%", "").strip()
        return float(cleaned) / 100.0
    v = float(s)
    if v > 1 and v <= 100:
        return v / 100.0
    return v

def parse_growth_profile(s, expect_years=None):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        out.append(parse_percent_or_decimal(p))
    if expect_years is not None and len(out) != expect_years:
        raise ValueError(f"Growth profile length {len(out)} != expected years {expect_years}")
    return out

# -----------------------
# DCF model
# -----------------------
def run_dcf(base_fcf, years, growth, wacc, perp_g, shares_out, growth_profile=None,
            base_fcf_is_t0=True, net_debt=0.0):
    growth_rates = growth_profile if growth_profile is not None else [growth] * years

    projections = []
    if base_fcf_is_t0:
        fcf = base_fcf
        for t in range(1, years+1):
            fcf = fcf * (1 + growth_rates[t-1])
            projections.append((t, fcf))
    else:
        fcf = base_fcf
        projections.append((1, fcf))
        for t in range(2, years+1):
            fcf = fcf * (1 + growth_rates[t-1])
            projections.append((t, fcf))

    proj_df = pd.DataFrame(projections, columns=["Year", "Projected_FCF"])
    proj_df["Discount_Factor"] = proj_df["Year"].apply(lambda t: (1 + wacc) ** t)
    proj_df["PV_FCF"] = proj_df["Projected_FCF"] / proj_df["Discount_Factor"]

    last_fcf = proj_df["Projected_FCF"].iloc[-1]
    if wacc <= perp_g:
        raise ValueError("WACC must be greater than perpetual growth rate for terminal value calculation.")
    tv = last_fcf * (1 + perp_g) / (wacc - perp_g)
    pv_tv = tv / ((1 + wacc) ** years)

    pv_sum = proj_df["PV_FCF"].sum()
    enterprise_value = pv_sum + pv_tv
    equity_value = enterprise_value - net_debt
    intrinsic_per_share = equity_value / shares_out

    return {
        "projection_df": proj_df.round(6),
        "pv_sum": pv_sum,
        "terminal_value": tv,
        "pv_terminal": pv_tv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_per_share": intrinsic_per_share
    }

# -----------------------
# Display helpers
# -----------------------
DISPLAY_DIVISORS = {
    "": 1.0,
    "raw": 1.0,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12
}

def pretty_currency(value):
    return f"₦{value:,.2f}"

def to_excel_bytes(proj_df, summary_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        proj_df.to_excel(writer, sheet_name='Projection', index=False)
        summary_df = pd.DataFrame(list(summary_dict.items()), columns=["Metric", "Value"])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    processed_data = output.getvalue()
    return processed_data


# -----------------------
# Streamlit UI
# -----------------------
st.title("Pride Advisory — DCF Valuation Tool")
st.markdown("Interactive DCF — enter values; the app will compute EV, Equity Value and intrinsic value per share.")

try:
    # Sidebar inputs
    st.sidebar.header("Inputs")
    base_fcf_input = st.sidebar.text_input("Base FCF (next year) — e.g., '550', '550m', '550 billion'", value="550bn")
    years = st.sidebar.number_input("Explicit forecast years", min_value=1, max_value=20, value=5, step=1)
    growth_mode = st.sidebar.radio("Growth input mode", ["Constant", "Per-year"])
    if growth_mode == "Constant":
        growth_input = st.sidebar.text_input("Annual growth (e.g., '8' or '8%')", "8%")
    else:
        growth_input = st.sidebar.text_input("Comma-separated per-year growth (e.g., '10%,8%,6%')", "")

    wacc_input = st.sidebar.text_input("WACC / Discount rate (e.g., '19%' or '0.19')", "19%")
    perp_g_input = st.sidebar.text_input("Perpetual growth (e.g., '3%')", "3%")
    shares_input = st.sidebar.text_input("Shares outstanding (e.g., '17bn' or '17000000000')", "17bn")
    net_debt_input = st.sidebar.text_input("Net debt (Debt - Cash), e.g., '850bn' or '0'", "850bn")
    base_fcf_t0 = st.sidebar.checkbox("Treat base FCF as t=0 (Year1 = t0*(1+g))", value=True)

    run = st.sidebar.button("Run DCF")

    # show note about blank page problems
    st.sidebar.markdown("---")
    st.sidebar.markdown("If the page is blank after deploy: check the app logs in Streamlit Cloud (Settings → Logs). Errors are shown there.")

    if run:
        # parse inputs
        base_fcf_val, b_unit = parse_money(base_fcf_input)
        if growth_mode == "Constant":
            growth = parse_percent_or_decimal(growth_input)
            growth_profile = None
        else:
            growth_profile = parse_growth_profile(growth_input, expect_years=years)
            growth = None

        wacc = parse_percent_or_decimal(wacc_input)
        perp_g = parse_percent_or_decimal(perp_g_input)
        shares_val, shares_unit = parse_money(shares_input)
        net_debt_val, net_debt_unit = parse_money(net_debt_input)

        # Basic sanity checks
        if wacc <= perp_g:
            st.error("WACC must be greater than perpetual growth rate.")
            st.stop()

        # Sanity: warn if net debt is HUGE compared to rough EV proxy
        approx_tv = base_fcf_val * ((1 + (growth if growth is not None else (growth_profile[0] if growth_profile else 0))) ** years) / max(1e-9, (wacc - perp_g))
        if net_debt_val > approx_tv * 3:
            st.warning("Net debt is much larger than a rough EV proxy — confirm units/inputs.")

        # If growth_profile provided use it otherwise use constant growth
        gp = growth_profile if growth_profile is not None else [growth] * years

        # run model
        res = run_dcf(
            base_fcf=base_fcf_val,
            years=years,
            growth=growth if growth is not None else gp[0],
            wacc=wacc,
            perp_g=perp_g,
            shares_out=shares_val,
            growth_profile=gp,
            base_fcf_is_t0=base_fcf_t0,
            net_debt=net_debt_val
        )

        proj_df = res["projection_df"]
        pv_sum = res["pv_sum"]
        tv = res["terminal_value"]
        pv_tv = res["pv_terminal"]
        ev_raw = res["enterprise_value"]
        equity_raw = res["equity_value"]
        per_share_raw = res["intrinsic_per_share"]

        # UI: show tables & results
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Projection and Present Values")
            st.dataframe(proj_df.style.format({"Projected_FCF":"{:.2f}", "PV_FCF":"{:.2f}"}), height=320)
            st.write(f"PV of explicit FCFs: {pv_sum:,.2f}")
            st.write(f"Terminal value (undiscounted): {tv:,.2f}")
            st.write(f"PV of Terminal value: {pv_tv:,.2f}")
            st.write(f"Enterprise Value (EV): {ev_raw:,.2f}")
        with col2:
            st.subheader("Equity & Per-share")
            st.write(f"Net debt (input): {net_debt_val:,.2f}")
            st.write(f"Equity value: {equity_raw:,.2f}")
            st.write(f"Intrinsic value per share: {per_share_raw:,.2f} ₦")
            st.download_button("Download results (Excel)",
                               to_excel_bytes(proj_df, {"EV":ev_raw,"Equity":equity_raw,"Per_share":per_share_raw}),
                               file_name="dcf_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

except Exception as e:
    st.error("An error occurred. See details below.")
    st.exception(e)
