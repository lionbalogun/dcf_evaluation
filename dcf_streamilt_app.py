# Simple DCF valuation tool
# Units are assumed consistent (e.g., ₦ billions). Shares_out should be in same unit (e.g., billions).
# Example: run_dcf(base_fcf=550, years=5, growth=0.08, wacc=0.19, perp_g=0.03, shares_out=17)

import math
import pandas as pd

def run_dcf(base_fcf, years, growth, wacc, perp_g, shares_out, growth_profile=None, show_tables=False):
    """
    base_fcf: float - FCF in the next year (e.g., 550)
    years: int - number of explicit forecast years (e.g., 5)
    growth: float - constant annual growth rate for the explicit period (e.g., 0.08)
    wacc: float - discount rate (e.g., 0.19)
    perp_g: float - perpetual growth rate used for terminal value (e.g., 0.03)
    shares_out: float - shares outstanding (same numeric units as EV numerator, e.g., 17)
    growth_profile: list or None - optional list of per-year growth rates (length == years)
    show_tables: bool - whether to print tables (useful in plain .py)
    """
    # Validate growth_profile if provided
    if growth_profile is not None and len(growth_profile) != years:
        raise ValueError("growth_profile length must equal number of years")

    # Build per-year growth rates
    growth_rates = growth_profile if growth_profile is not None else [growth] * years

    # Project FCFs
    projections = []
    fcf = base_fcf
    for t in range(1, years + 1):
        fcf = fcf * (1 + growth_rates[t-1])
        projections.append((t, round(fcf, 6)))

    proj_df = pd.DataFrame(projections, columns=["Year", "Projected_FCF"])

    # Discount factors and PVs
    proj_df["Discount_Factor"] = proj_df["Year"].apply(lambda t: (1 + wacc) ** t)
    proj_df["PV_FCF"] = proj_df["Projected_FCF"] / proj_df["Discount_Factor"]
    proj_df["PV_FCF"] = proj_df["PV_FCF"].round(6)

    # Terminal value based on last projected FCF
    last_fcf = proj_df["Projected_FCF"].iloc[-1]
    if wacc <= perp_g:
        raise ValueError("WACC must be greater than perpetual growth rate for terminal value calculation.")
    tv = last_fcf * (1 + perp_g) / (wacc - perp_g)
    pv_tv = tv / ((1 + wacc) ** years)

    pv_sum = proj_df["PV_FCF"].sum()
    enterprise_value = round(pv_sum + pv_tv, 6)
    intrinsic_per_share = round(enterprise_value / shares_out, 6)

    results = {
        "projection_table": proj_df,
        "pv_sum": round(pv_sum, 6),
        "terminal_value": round(tv, 6),
        "pv_terminal": round(pv_tv, 6),
        "enterprise_value": enterprise_value,
        "intrinsic_per_share": intrinsic_per_share,
        "inputs": {
            "base_fcf": base_fcf,
            "years": years,
            "growth": growth,
            "wacc": wacc,
            "perp_g": perp_g,
            "shares_out": shares_out
        }
    }

    # Sensitivity sample table (WACC +/- and perp_g +/-)
    wacc_range = [round(wacc - 0.04, 6), round(wacc - 0.02, 6), round(wacc, 6), round(wacc + 0.02, 6)]
    g_range = [round(perp_g - 0.01, 6), round(perp_g, 6), round(perp_g + 0.01, 6)]

    sens_rows = []
    for wa in wacc_range:
        for g in g_range:
            if wa <= g:
                val = None
            else:
                tv_s = last_fcf * (1 + g) / (wa - g)
                pv_tv_s = tv_s / ((1 + wa) ** years)
                ev_s = round(pv_sum + pv_tv_s, 6)
                val = round(ev_s / shares_out, 6)
            sens_rows.append({"WACC": wa, "Perp_G": g, "Intrinsic_per_share": val})

    sens_df = pd.DataFrame(sens_rows)

    if show_tables:
        print("\nInputs:")
        print(pd.DataFrame([results["inputs"]]).T)
        print("\nProjection table:")
        print(proj_df.to_string(index=False))
        print(f"\nPV of explicit FCFs: {results['pv_sum']}")
        print(f"Terminal Value (undiscounted): {results['terminal_value']}")
        print(f"PV of Terminal Value: {results['pv_terminal']}")
        print(f"Enterprise Value: {results['enterprise_value']}")
        print(f"Intrinsic Value per Share: {results['intrinsic_per_share']}")
        print("\nSensitivity table (sample):")
        print(sens_df.to_string(index=False))

    results["sensitivity_table"] = sens_df
    return results

# Example usage
if __name__ == "__main__":
    res = run_dcf(base_fcf=550, years=5, growth=0.08, wacc=0.19, perp_g=0.03, shares_out=17, show_tables=True)
