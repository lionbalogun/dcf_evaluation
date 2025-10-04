#!/usr/bin/env python3
"""
Interactive DCF CLI — updated with sanity checks and clearer output formatting.
Save as dcf_strea3.py and run with: python dcf_strea3.py
"""
import re
import argparse
import sys
import math
import pandas as pd

# -----------------------
# Parsing helpers
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
            show_tables=True, base_fcf_is_t0=True, net_debt=0.0, unit_name=""):
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

    results = {
        "projection_table": proj_df.round(6),
        "pv_sum": round(pv_sum, 6),
        "terminal_value": round(tv, 6),
        "pv_terminal": round(pv_tv, 6),
        "enterprise_value": round(enterprise_value, 6),
        "equity_value": round(equity_value, 6),
        "intrinsic_per_share": round(intrinsic_per_share, 6),
        "unit_name": unit_name
    }

    if show_tables:
        print("\nProjection table (units: {}):".format(unit_name or "base units"))
        print(results["projection_table"].to_string(index=False))
        print(f"\nPV of explicit FCFs ({unit_name}): {results['pv_sum']}")
        print(f"Terminal Value (undiscounted) ({unit_name}): {results['terminal_value']}")
        print(f"PV of Terminal Value ({unit_name}): {results['pv_terminal']}")
        print(f"Enterprise Value ({unit_name}): {results['enterprise_value']}")
        print(f"Equity Value (EV - net_debt) ({unit_name}): {results['equity_value']}")
        print(f"Intrinsic Value per Share ({unit_name} per share): {results['intrinsic_per_share']}")
    return results

# -----------------------
# Interactive prompts
# -----------------------
def prompt_loop(prompt_text, parse_fn, default=None, validator=None, help_text=None):
    while True:
        default_hint = f" [default={default}]" if default is not None else ""
        if help_text:
            print(help_text)
        raw = input(f"{prompt_text}{default_hint}: ").strip()
        if raw == "" and default is not None:
            raw = str(default)
        try:
            val = parse_fn(raw)
            if validator and not validator(val):
                print("Validation failed. Try again.")
                continue
            return val
        except Exception as e:
            print("Invalid input:", e)
            print("Please try again.\n")

def ask_money_with_unit(prompt_text, default_value=None):
    def pfunc(raw):
        v, unit = parse_money(raw)
        unit_name = ""
        if unit == "k": unit_name = "thousand"
        elif unit in ("m", "mn"): unit_name = "million"
        elif unit in ("b", "bn"): unit_name = "billion"
        elif unit in ("t", "tr","tn"): unit_name = "trillion"
        return v, unit_name
    return prompt_loop(prompt_text, pfunc, default=default_value)

def ask_shares(prompt_text, default_value=None):
    def pfunc(raw):
        v, unit = parse_money(raw)
        return v, unit
    return prompt_loop(prompt_text, pfunc, default=default_value)

# -----------------------
# Pretty display helpers
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

def format_money_for_display(value, unit_label):
    label = unit_label or "raw"
    divisor = DISPLAY_DIVISORS.get(label, 1.0)
    return value / divisor, label

# -----------------------
# Main CLI / interactive flow
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Friendly DCF CLI")
    parser.add_argument("--noninteractive", action="store_true", help="(unused)")
    args = parser.parse_args()

    print("=== Interactive DCF — type values like '550m', '5 billion', '19%', '0.19' ===\n")

    base_fcf_val, base_fcf_unit = ask_money_with_unit("Base FCF (e.g., '550m' or '5bn')", default_value="550")
    years = prompt_loop("Forecast years (int)", lambda s: int(s), default="5", validator=lambda x: x > 0)
    growth = prompt_loop("Constant growth (e.g., '8%' or 0.08)", parse_percent_or_decimal, default="0.08")
    gp_raw = input("Optional growth profile (comma-separated rates, e.g. '10%,8%,6%') [leave blank to use constant growth]: ").strip()
    growth_profile = None
    if gp_raw != "":
        try:
            growth_profile = parse_growth_profile(gp_raw, expect_years=years)
        except Exception as e:
            print("Could not parse growth profile:", e)
            print("Falling back to constant growth.\n")
            growth_profile = None

    wacc = prompt_loop("WACC / discount rate (e.g. '19%' or 0.19)", parse_percent_or_decimal, default="0.19", validator=lambda x: x >= 0 and x < 10)
    perp_g = prompt_loop("Perpetual growth (e.g. '3%' or 0.03)", parse_percent_or_decimal, default="0.03", validator=lambda x: x >= -1 and x < 1)

    while not (wacc > perp_g):
        print(f"\nWACC must be greater than perpetual growth rate for terminal value calculation.")
        wacc = prompt_loop("WACC / discount rate (e.g. '19%' or 0.19)", parse_percent_or_decimal, default=str(wacc), validator=lambda x: x >= 0 and x < 10)
        perp_g = prompt_loop("Perpetual growth (e.g. '3%' or 0.03)", parse_percent_or_decimal, default=str(perp_g), validator=lambda x: x >= -1 and x < 1)

    print("\nShares outstanding: it's important to use the SAME unit as your FCF to get correct per-share values.")
    print(f"Your FCF input looked like: {base_fcf_val} (unit: '{base_fcf_unit or 'raw/none'}').")
    shares_val, shares_unit = ask_shares("Shares outstanding (e.g., '17', '17m', '17 billion')", default_value="17")

    # Ask which unit for per-share display
    unit_choice = ""
    if base_fcf_unit:
        choice = input(f"Return per-share value in the same unit as FCF ('{base_fcf_unit}')? (Y/n): ").strip().lower()
        if choice in ("", "y", "yes"):
            unit_choice = base_fcf_unit
    if unit_choice == "":
        unit_choice = input("Which unit should per-share be reported in? (blank=raw, options: thousand,million,billion,trillion): ").strip().lower()

    shares_abs = shares_val  # parse_money returned absolute number if units included

    net_debt_val, net_debt_unit = ask_money_with_unit("Net debt (debt - cash), e.g., '20m' or '0' (will be subtracted from EV)", default_value="0")
    net_debt_abs = net_debt_val

    b_t0_raw = input("Is the Base FCF you entered the t=0 (current year) FCF so that Year1 = t0*(1+g)? (y/N): ").strip().lower()
    base_fcf_is_t0 = b_t0_raw in ("y", "yes")

    # Sanity check: rough EV proxy vs net debt
    try:
        approx_tv = base_fcf_val * ((1 + growth) ** years) / max(1e-9, (wacc - perp_g))
        approx_ev_proxy = approx_tv
        if net_debt_abs > approx_ev_proxy * 3:   # if net debt > ~3x rough EV proxy, warn and confirm
            print("\nWARNING: Net debt is much larger than the rough EV proxy based on your inputs.")
            print(f"Rough EV proxy (approx): {approx_ev_proxy:.3e}, Net debt: {net_debt_abs:.3e}")
            confirm = input("Proceed anyway? (y/N): ").strip().lower()
            if confirm not in ('y','yes'):
                print("Aborting — please re-check your units/inputs.")
                sys.exit(0)
    except Exception:
        pass

    print("\nRunning DCF with these interpreted values (absolute currency numbers shown):")
    print(f"Base FCF absolute: {base_fcf_val}")
    print(f"Forecast years: {years}")
    print(f"Growth (annual): {growth:.6f}")
    print(f"WACC: {wacc:.6f}")
    print(f"Perpetual growth: {perp_g:.6f}")
    print(f"Shares outstanding (absolute): {shares_abs}")
    print(f"Net debt (absolute): {net_debt_abs}")
    print(f"Base FCF treated as t=0? {base_fcf_is_t0}")
    print("Unit label (for per-share results):", unit_choice or "raw (no label)")
    print("-------------------------------------------------------------------\n")

    # Run model
    try:
        res = run_dcf(
            base_fcf=base_fcf_val,
            years=years,
            growth=growth,
            wacc=wacc,
            perp_g=perp_g,
            shares_out=shares_abs,
            growth_profile=growth_profile,
            show_tables=False,
            base_fcf_is_t0=base_fcf_is_t0,
            net_debt=net_debt_abs,
            unit_name=unit_choice or "raw"
        )
    except Exception as e:
        print("Error running DCF:", e)
        sys.exit(1)

    # Pretty display: EV/Equity scaled, per-share in ₦
    ev_raw = res["enterprise_value"]
    equity_raw = res["equity_value"]
    per_share_raw = equity_raw / shares_abs

    display_unit = unit_choice or ""
    div = DISPLAY_DIVISORS.get(display_unit, 1.0)
    unit_label = display_unit if display_unit != "" else "raw"

    ev_scaled = ev_raw / div
    equity_scaled = equity_raw / div

    print("\n--- DCF Results (human-friendly) ---")
    print(f"Enterprise Value (scaled): {ev_scaled:,.3f} {unit_label} (raw: {pretty_currency(ev_raw)})")
    print(f"Equity Value (scaled):     {equity_scaled:,.3f} {unit_label} (raw: {pretty_currency(equity_raw)})")
    print(f"Intrinsic value per share: {pretty_currency(per_share_raw)} per share")
    print("------------------------------------\n")

if __name__ == "__main__":
    main()
