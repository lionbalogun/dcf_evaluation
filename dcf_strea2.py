#!/usr/bin/env python3
"""
Interactive / CLI DCF tool.

Usage (interactive):
    python dcf_tool.py

Usage (CLI):
    python dcf_tool.py --base_fcf 550 --years 5 --growth 0.08 --wacc 0.19 --perp_g 0.03 --shares_out 17

You can also pass --growth_profile "0.1,0.09,0.08,0.07,0.06" (must match years).
"""
import argparse
import math
import pandas as pd
import sys

def run_dcf(base_fcf, years, growth, wacc, perp_g, shares_out,
            growth_profile=None, show_tables=False,
            base_fcf_is_t0=True, net_debt=0.0):
    # Validate growth_profile if provided
    if growth_profile is not None and len(growth_profile) != years:
        raise ValueError("growth_profile length must equal number of years")

    growth_rates = growth_profile if growth_profile is not None else [growth] * years

    # Project FCFs
    projections = []
    if base_fcf_is_t0:
        fcf = base_fcf
        for t in range(1, years + 1):
            fcf = fcf * (1 + growth_rates[t-1])
            projections.append((t, fcf))
    else:
        fcf = base_fcf
        projections.append((1, fcf))
        for t in range(2, years + 1):
            fcf = fcf * (1 + growth_rates[t-1])
            projections.append((t, fcf))

    proj_df = pd.DataFrame(projections, columns=["Year", "Projected_FCF"])

    # Discount explicit FCFs using provided wacc
    proj_df["Discount_Factor"] = proj_df["Year"].apply(lambda t: (1 + wacc) ** t)
    proj_df["PV_FCF"] = proj_df["Projected_FCF"] / proj_df["Discount_Factor"]

    # Terminal value
    last_fcf = proj_df["Projected_FCF"].iloc[-1]
    if wacc <= perp_g:
        raise ValueError("WACC must be greater than perpetual growth rate for terminal value calculation.")
    tv = last_fcf * (1 + perp_g) / (wacc - perp_g)
    pv_tv = tv / ((1 + wacc) ** years)

    pv_sum = proj_df["PV_FCF"].sum()
    enterprise_value = pv_sum + pv_tv
    equity_value = enterprise_value - net_debt
    intrinsic_per_share = equity_value / shares_out

    # Sensitivity: recompute PV of explicit FCFs for each WACC
    wacc_range = [round(wacc - 0.04, 6), round(wacc - 0.02, 6), round(wacc, 6), round(wacc + 0.02, 6)]
    g_range = [round(perp_g - 0.01, 6), round(perp_g, 6), round(perp_g + 0.01, 6)]

    sens_rows = []
    for wa in wacc_range:
        for g in g_range:
            if wa <= g:
                val = None
            else:
                pv_explicit = 0.0
                for _, row in proj_df.iterrows():
                    t = int(row["Year"])
                    f = float(row["Projected_FCF"])
                    pv_explicit += f / ((1 + wa) ** t)
                tv_s = last_fcf * (1 + g) / (wa - g)
                pv_tv_s = tv_s / ((1 + wa) ** years)
                ev_s = pv_explicit + pv_tv_s
                equity_value_s = ev_s - net_debt
                val = equity_value_s / shares_out
            sens_rows.append({"WACC": wa, "Perp_G": g, "Intrinsic_per_share": None if val is None else round(val, 6)})

    sens_df = pd.DataFrame(sens_rows)

    results = {
        "projection_table": proj_df.round(6),
        "pv_sum": round(pv_sum, 6),
        "terminal_value": round(tv, 6),
        "pv_terminal": round(pv_tv, 6),
        "enterprise_value": round(enterprise_value, 6),
        "equity_value": round(equity_value, 6),
        "intrinsic_per_share": round(intrinsic_per_share, 6),
        "sensitivity_table": sens_df,
        "inputs": {
            "base_fcf": base_fcf,
            "years": years,
            "growth": growth,
            "wacc": wacc,
            "perp_g": perp_g,
            "shares_out": shares_out,
            "base_fcf_is_t0": base_fcf_is_t0,
            "net_debt": net_debt
        }
    }

    if show_tables:
        print("\nInputs:")
        print(pd.DataFrame([results["inputs"]]).T)
        print("\nProjection table:")
        print(results["projection_table"].to_string(index=False))
        print(f"\nPV of explicit FCFs: {results['pv_sum']}")
        print(f"Terminal Value (undiscounted): {results['terminal_value']}")
        print(f"PV of Terminal Value: {results['pv_terminal']}")
        print(f"Enterprise Value: {results['enterprise_value']}")
        print(f"Equity Value (EV - net_debt): {results['equity_value']}")
        print(f"Intrinsic Value per Share: {results['intrinsic_per_share']}")
        print("\nSensitivity table (sample):")
        print(sens_df.to_string(index=False))

    results["sensitivity_table"] = sens_df
    return results

def parse_growth_profile(s):
    """Parse a comma-separated list of growth rates into floats (e.g. '0.1,0.08,0.06')."""
    if s is None or s.strip() == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("Growth profile must be comma-separated floats, e.g. 0.1,0.08,0.06")
    return vals

def interactive_input(prompt, cast=float, default=None, validator=None):
    while True:
        raw = input(f"{prompt}" + (f" [default={default}]" if default is not None else "") + ": ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = cast(raw)
            if validator and not validator(val):
                print("Invalid value; failed validation.")
                continue
            return val
        except Exception as e:
            print("Invalid input:", e)

def main():
    parser = argparse.ArgumentParser(description="Simple DCF CLI + Interactive tool")
    parser.add_argument("--base_fcf", type=float, help="Base FCF (if base_fcf_is_t0=True this is t=0 FCF).")
    parser.add_argument("--years", type=int, help="Explicit forecast years (int).")
    parser.add_argument("--growth", type=float, help="Constant annual growth (if growth_profile not used).")
    parser.add_argument("--wacc", type=float, help="Discount rate (WACC).")
    parser.add_argument("--perp_g", type=float, help="Perpetual growth rate for terminal value.")
    parser.add_argument("--shares_out", type=float, help="Shares outstanding (same units as EV numerator).")
    parser.add_argument("--growth_profile", type=parse_growth_profile,
                        help="Optional comma-separated per-year growth rates, e.g. '0.1,0.08,0.06'.")
    parser.add_argument("--net_debt", type=float, default=0.0, help="Net debt (debt - cash).")
    parser.add_argument("--base_fcf_is_t0", action="store_true", help="If supplied, treat base_fcf as t=0 (default False if not supplied).")
    parser.add_argument("--show_tables", action="store_true", help="Print tables for inspection.")
    args = parser.parse_args()

    # If no CLI args were provided at all, launch interactive prompts
    if len(sys.argv) == 1:
        print("Interactive DCF inputs (press Enter to accept defaults shown):")
        base_fcf = interactive_input("Base FCF (numeric)", cast=float, default=550.0)
        years = interactive_input("Forecast years (int)", cast=int, default=5, validator=lambda x: x > 0)
        growth = interactive_input("Constant growth (e.g., 0.08)", cast=float, default=0.08)
        wacc = interactive_input("WACC (e.g., 0.19)", cast=float, default=0.19)
        perp_g = interactive_input("Perpetual growth (e.g., 0.03)", cast=float, default=0.03)
        shares_out = interactive_input("Shares outstanding (same units as EV numerator)", cast=float, default=17.0)
        gp_raw = input("Optional growth profile (comma-separated, leave blank to use constant growth): ").strip()
        growth_profile = parse_growth_profile(gp_raw) if gp_raw else None
        net_debt = interactive_input("Net debt (debt - cash), default 0", cast=float, default=0.0)
        base_fcf_is_t0 = input("Is base_fcf t=0 (current year)? (y/N): ").strip().lower() in ("y", "yes")
        show_tables = True
    else:
        # Use CLI args where provided; otherwise prompt minimally for missing required args
        def get_or_prompt(attr, prompt_text, cast, default=None, required=True):
            val = getattr(args, attr)
            if val is not None:
                return val
            if required:
                return interactive_input(prompt_text, cast=cast, default=default)
            return default

        base_fcf = get_or_prompt("base_fcf", "Base FCF (numeric)", float, default=550.0)
        years = get_or_prompt("years", "Forecast years (int)", int, default=5)
        growth = get_or_prompt("growth", "Constant growth (e.g., 0.08)", float, default=0.08)
        wacc = get_or_prompt("wacc", "WACC (e.g., 0.19)", float, default=0.19)
        perp_g = get_or_prompt("perp_g", "Perpetual growth (e.g., 0.03)", float, default=0.03)
        shares_out = get_or_prompt("shares_out", "Shares outstanding", float, default=17.0)
        growth_profile = args.growth_profile
        net_debt = args.net_debt
        base_fcf_is_t0 = args.base_fcf_is_t0
        show_tables = args.show_tables

    # Run model
    try:
        res = run_dcf(
            base_fcf=base_fcf,
            years=years,
            growth=growth,
            wacc=wacc,
            perp_g=perp_g,
            shares_out=shares_out,
            growth_profile=growth_profile,
            show_tables=show_tables,
            base_fcf_is_t0=base_fcf_is_t0,
            net_debt=net_debt
        )
        # If show_tables False, still print summary
        if not show_tables:
            print("\nIntrinsic value per share:", res["intrinsic_per_share"])
    except Exception as e:
        print("Error running DCF:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
