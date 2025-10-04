#!/usr/bin/env python3
"""
Geek-friendly interactive DCF CLI.

Features:
 - Parse money inputs like: "550", "550 million", "550m", "₦550m", "5bn", "5 billion", "5,000,000"
 - Parse rates like: "19%", "0.19", "19"
 - Parse growth profiles like: "10%,8%,6%" or "0.10,0.08,0.06"
 - Validate WACC > perp_g; re-prompt until valid
 - Keep units consistent between FCF and shares (asks and normalizes)
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
    """
    Parse human-friendly monetary strings into a float (base unit: 1.0).
    Examples:
      "550" -> 550.0
      "550m" -> 550_000_000.0
      "5 billion" -> 5_000_000_000.0
      "₦5bn" -> 5_000_000_000.0
      "5,000,000" -> 5000000.0
    Returns (value_in_base_unit, detected_unit_multiplier_name)
    """
    if s is None:
        return None, ""
    s = str(s).strip().lower()
    if s == "":
        return None, ""
    # Remove currency symbols and spaces around commas
    s = s.replace("₦", "").replace("ngn", "").replace("$", "")
    s = s.replace(",", "")
    # detect pattern: number + optional unit word/letter
    # e.g. "5.5bn", "500m", "5 billion", "500000"
    m = re.match(r"^(-?\d+(\.\d+)?)(\s*([a-zA-Z]+))?$", s)
    if m:
        num = float(m.group(1))
        unit_token = m.group(4) or ""
        unit_token = unit_token.strip().lower()
        # map common abbreviations: "million" -> "m", etc
        # if unit contains percent sign etc, return None
        unit = ""
        for token, mult in UNIT_MULTIPLIERS.items():
            if token != "" and (unit_token == token or unit_token.startswith(token)):
                unit = token
                multiplier = mult
                return num * multiplier, unit
        # If no unit matched, try some heuristics:
        # - tokens like "m" or "mn" included above; else assume raw number
        return num, ""
    # fallback: user wrote "5 million" with a space and extra char?
    for word, mult in UNIT_MULTIPLIERS.items():
        if word != "" and word in s:
            # remove the word and any non-numeric chars
            cleaned = re.sub(r"[^\d\.\-]", "", s)
            if cleaned == "":
                continue
            try:
                num = float(cleaned)
                return num * mult, word
            except:
                pass
    # final fallback: try to parse as plain float after removing non-digit chars
    cleaned = re.sub(r"[^\d\.\-]", "", s)
    if cleaned:
        try:
            return float(cleaned), ""
        except:
            pass
    raise ValueError(f"Could not parse money value: '{s}'")

def parse_percent_or_decimal(s):
    """
    Parse percent or decimal:
     - "19%" -> 0.19
     - "0.19" -> 0.19
     - "19" -> 0.19 (assume if >=1 and <=100 treat as percent)
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    # accept something like "19 %"
    if "%" in s:
        cleaned = s.replace("%", "").strip()
        try:
            return float(cleaned) / 100.0
        except:
            raise ValueError(f"Could not parse percentage: '{s}'")
    # else plain float
    try:
        v = float(s)
        if v > 1 and v <= 100:  # assume user wrote "19" meaning 19%
            return v / 100.0
        return v  # already decimal (e.g., 0.19 or 0.019)
    except:
        raise ValueError(f"Could not parse decimal/percent: '{s}'")

def parse_growth_profile(s, expect_years=None):
    """
    Accept comma-separated growth inputs which can be percentages or decimals.
    e.g. "10%,8%,6%" or "0.10,0.08,0.06"
    Returns list of floats (decimals)
    """
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        # allow entries like "100m" only if expect_years and it's a money series; but for now we expect rates
        try:
            out.append(parse_percent_or_decimal(p))
        except ValueError:
            # try parse_money -> not typical; raise because growth profile should be rates
            raise ValueError(f"Growth profile elements must be percents or decimals, failed at '{p}'")
    if expect_years is not None and len(out) != expect_years:
        raise ValueError(f"Growth profile length {len(out)} != expected years {expect_years}")
    return out

# -----------------------
# DCF model (same as improved version)
# -----------------------
def run_dcf(base_fcf, years, growth, wacc, perp_g, shares_out, growth_profile=None,
            show_tables=True, base_fcf_is_t0=True, net_debt=0.0, unit_name=""):
    # growth_profile validated by caller
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
    """
    Repeatedly prompt until parse_fn succeeds and validator returns True (if provided).
    parse_fn must raise ValueError on invalid parse.
    """
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
    """
    Ask money input and return (value_in_base_units, unit_label)
    Unit label is one of '', 'thousand', 'million', 'billion', 'trillion'
    """
    def pfunc(raw):
        v, unit = parse_money(raw)
        # convert unit token to friendly name
        unit_name = ""
        if unit == "k": unit_name = "thousand"
        elif unit in ("m", "mn"): unit_name = "million"
        elif unit in ("b", "bn"): unit_name = "billion"
        elif unit in ("t", "tr","tn"): unit_name = "trillion"
        return v, unit_name
    return prompt_loop(prompt_text, pfunc, default=default_value)

def ask_shares(prompt_text, default_value=None, unit_reference=None):
    """
    Accept shares input, allow values like '17', '17m', '17 million', or raw number.
    If unit_reference is provided (e.g., 'billion'), will ask shares in same unit or convert.
    Returns (shares_numeric_in_same_base_as_fcf, unit_label)
    """
    def pfunc(raw):
        v, unit = parse_money(raw)
        # if unit is '', interpret as raw count (e.g., 17 -> 17 shares)
        # We'll return shares in same "base" as FCF later by normalizing
        return v, unit
    return prompt_loop(prompt_text, pfunc, default=default_value)

# -----------------------
# Main CLI / interactive flow
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Friendly DCF CLI")
    parser.add_argument("--noninteractive", action="store_true", help="Exit after printing usage (not used).")
    args = parser.parse_args()

    print("=== Interactive DCF — type values like '550m', '5 billion', '19%', '0.19' ===\n")

    # Base FCF
    base_fcf_val, base_fcf_unit = ask_money_with_unit("Base FCF (e.g., '550m' or '5bn')", default_value="550")
    # years
    years = prompt_loop("Forecast years (int)", lambda s: int(s), default="5", validator=lambda x: x > 0)
    # growth (can be percent or decimal)
    growth = prompt_loop("Constant growth (e.g., '8%' or 0.08)", parse_percent_or_decimal, default="0.08")
    # allow optional growth profile (rates)
    gp_raw = input("Optional growth profile (comma-separated rates, e.g. '10%,8%,6%') [leave blank to use constant growth]: ").strip()
    growth_profile = None
    if gp_raw != "":
        try:
            growth_profile = parse_growth_profile(gp_raw, expect_years=years)
        except Exception as e:
            print("Could not parse growth profile:", e)
            print("Falling back to constant growth.\n")
            growth_profile = None

    # WACC and perp_g must be asked and validated to ensure wacc > perp_g
    def wacc_validator(x):
        # we'll validate relationship after reading perp_g; here accept any numeric between 0 and 1
        return x >= 0 and x < 10

    wacc = prompt_loop("WACC / discount rate (e.g. '19%' or 0.19)", parse_percent_or_decimal, default="0.19", validator=wacc_validator)
    perp_g = prompt_loop("Perpetual growth (e.g. '3%' or 0.03)", parse_percent_or_decimal, default="0.03", validator=lambda x: x >= -1 and x < 1)

    # Validate relational constraint
    while not (wacc > perp_g):
        print(f"\nWACC must be greater than perpetual growth rate for terminal value calculation.")
        print(f"You entered WACC={wacc:.6f} and perp_g={perp_g:.6f}. Please re-enter.\n")
        wacc = prompt_loop("WACC / discount rate (e.g. '19%' or 0.19)", parse_percent_or_decimal, default=str(wacc), validator=wacc_validator)
        perp_g = prompt_loop("Perpetual growth (e.g. '3%' or 0.03)", parse_percent_or_decimal, default=str(perp_g), validator=lambda x: x >= -1 and x < 1)

    # Shares outstanding — encourage same unit as FCF
    print("\nShares outstanding: it's important to use the SAME unit as your FCF to get correct per-share values.")
    print(f"Your FCF input looked like: {base_fcf_val} (unit: '{base_fcf_unit or 'raw/none'}').")
    shares_val, shares_unit = ask_shares("Shares outstanding (e.g., '17', '17m', '17 million')", default_value="17")

    # Normalize units: choose base unit as raw number (no multiplier). We'll convert shares so that per-share uses same unit base.
    # Easiest approach: Ask user what unit they'd like per-share returned in.
    unit_choice = ""
    if base_fcf_unit:
        # propose using same unit as base_fcf
        choice = input(f"Return per-share value in the same unit as FCF ('{base_fcf_unit}')? (Y/n): ").strip().lower()
        if choice in ("", "y", "yes"):
            unit_choice = base_fcf_unit
    if unit_choice == "":
        # ask explicitly
        unit_choice = input("Which unit should per-share be reported in? (blank=raw, options: thousand,million,billion,trillion): ").strip().lower()

    # Convert everything to chosen base for display consistency:
    # We'll compute in raw currency units (e.g., naira), but tracking user's chosen display unit.
    # parse_money returned numbers in absolute units (e.g., 550m -> 550_000_000)
    # shares_val might be given as 17 or 17m. We need to convert to an absolute number of shares:
    shares_abs = shares_val  # parse_money already returns absolute number if unit present, else raw number
    if shares_unit in ("m","mn","million"):
        shares_abs = shares_val
    # if shares were provided like '17' we assume it's 17 (not 17m). That's fine if user intended same unit.

    # Ask net_debt
    net_debt_val, net_debt_unit = ask_money_with_unit("Net debt (debt - cash), e.g., '20m' or '0' (will be subtracted from EV)", default_value="0")
    net_debt_abs = net_debt_val

    # Ask base_fcf convention: t=0 or Year1
    b_t0_raw = input("Is the Base FCF you entered the t=0 (current year) FCF so that Year1 = t0*(1+g)? (y/N): ").strip().lower()
    base_fcf_is_t0 = b_t0_raw in ("y", "yes")

    # For run_dcf we want:
    # - base_fcf absolute number (parse_money gave absolute)
    # - shares_out should be absolute number of shares.
    # But many users think in 'billions' shares and 'billions' currency; to preserve their simplicity we'll let them keep numbers as-is.
    # If user entered shares like '17m' parse_money returned 17,000,000.
    # Good.

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
            show_tables=True,
            base_fcf_is_t0=base_fcf_is_t0,
            net_debt=net_debt_abs,
            unit_name=unit_choice or "raw"
        )
    except Exception as e:
        print("Error running DCF:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
