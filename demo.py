"""
Oil Derivatives Pricer — Interactive Demo
==========================================
Run this file to see the pricer in action with
realistic WTI crude oil market parameters.
"""

from pricer import OilDerivativesPricer, OilMarketData, OptionParams
import numpy as np


def separator(title=""):
    print("\n" + "═" * 60)
    if title:
        print(f"  {title}")
        print("═" * 60)


def main():
    print("\n" + "█" * 60)
    print("  OIL DERIVATIVES PRICER — WTI CRUDE OIL")
    print("█" * 60)

    # ── Market Setup ──────────────────────────────────────────
    # Realistic WTI crude oil parameters
    market = OilMarketData(
        spot_price=82.50,        # $/barrel
        risk_free_rate=0.0525,   # Fed funds ~5.25%
        convenience_yield=0.035, # crude oil convenience yield
        storage_cost=0.015,      # Cushing storage ~1.5%/yr
        volatility=0.32          # WTI vol ~32%
    )

    pricer = OilDerivativesPricer(market)

    print(f"\n  Spot Price:          ${market.spot_price:.2f}/bbl")
    print(f"  Risk-Free Rate:      {market.risk_free_rate*100:.2f}%")
    print(f"  Convenience Yield:   {market.convenience_yield*100:.2f}%")
    print(f"  Storage Cost:        {market.storage_cost*100:.2f}%")
    print(f"  Volatility (σ):      {market.volatility*100:.1f}%")
    print(f"  Cost of Carry (b):   {market.cost_of_carry*100:.2f}%")


    # ── 1. FUTURES CURVE ─────────────────────────────────────
    separator("1. FUTURES TERM STRUCTURE")
    curve = pricer.futures_curve(maturity=1.0)
    print(f"\n  Front month to 2Y futures prices (Cost-of-Carry model)")
    print(f"  F = S × exp(b × T)  where b = r + storage - convenience")
    print()
    for tenor, price in curve["term_structure"].items():
        bar = "█" * int((price - 70) / 1)
        print(f"  {tenor:>4}:  ${price:>7.2f}  {bar}")


    # ── 2. EUROPEAN OPTIONS — Black-76 ────────────────────────
    separator("2. EUROPEAN OPTIONS — BLACK-76 MODEL")
    print("\n  Industry-standard model for options on futures\n")

    strike = 82.50  # ATM
    maturity = 0.5  # 6 months

    for opt_type in ['call', 'put']:
        result = pricer.price_option(strike, maturity, opt_type, method='black76')
        print(f"  {opt_type.upper():4s}  K=${strike}  T={maturity}Y  →  ${result.price:.4f}")
        g = result.greeks
        print(f"         Δ={g['delta']:+.4f}  Γ={g['gamma']:.6f}  "
              f"ν={g['vega']:.4f}  Θ={g['theta']:.4f}")
        print()


    # ── 3. METHOD COMPARISON ─────────────────────────────────
    separator("3. METHOD COMPARISON (ATM 6M Call)")
    summary = pricer.summary(strike=82.50, maturity=0.5, option_type='call')

    print(f"\n  {summary['option']}\n")
    for method, price in summary["prices"].items():
        print(f"  {method:<35}  ${price:.4f}")

    prem = summary['early_exercise_premium']
    print(f"\n  Early Exercise Premium (American - European):  ${prem:.4f}")
    print("  (Low for calls on non-dividend paying futures)")


    # ── 4. STRIKE PROFILE ────────────────────────────────────
    separator("4. OPTION PRICES ACROSS STRIKES (6M, Black-76)")
    strikes = [60, 65, 70, 75, 80, 82.5, 85, 90, 95, 100, 110]
    print(f"\n  {'Strike':>8}  {'Moneyness':>10}  {'Call':>8}  {'Put':>8}  {'Delta':>8}")
    print("  " + "-" * 50)
    for K in strikes:
        call = pricer.price_option(K, 0.5, 'call')
        put  = pricer.price_option(K, 0.5, 'put')
        mness = "ATM" if K == 82.5 else ("ITM" if K < 82.5 else "OTM")
        print(f"  ${K:>6.1f}  {mness:>10}  ${call.price:>6.2f}  "
              f"${put.price:>6.2f}  {call.greeks['delta']:>+8.4f}")


    # ── 5. ASIAN OPTIONS ─────────────────────────────────────
    separator("5. ASIAN OPTIONS (Monte Carlo, 50k paths)")
    print("\n  Asian options widely used in oil markets for")
    print("  physical delivery contracts (MOPS, Platts average pricing)\n")

    asian_arith = pricer.price_asian(82.50, 0.5, 'call', 'arithmetic')
    asian_geom  = pricer.price_asian(82.50, 0.5, 'call', 'geometric')
    european_mc = pricer.price_option(82.50, 0.5, 'call', method='monte_carlo')

    print(f"  European Call (MC):        ${european_mc.price:.4f}")
    print(f"  Asian Call (Arithmetic):   ${asian_arith.price:.4f}  "
          f"  ±{asian_arith.metadata['std_error']:.4f}")
    print(f"  Asian Call (Geometric):    ${asian_geom.price:.4f}  "
          f"  ±{asian_geom.metadata['std_error']:.4f}")
    print(f"\n  Asian discount vs European: "
          f"${european_mc.price - asian_arith.price:.4f} "
          f"({(1 - asian_arith.price/european_mc.price)*100:.1f}%)")


    # ── 6. BARRIER OPTIONS ────────────────────────────────────
    separator("6. BARRIER OPTIONS (Monte Carlo)")
    print("\n  Barrier options reduce hedging costs — common")
    print("  structure: producer buys down-and-out put (knock-out floor)\n")

    vanilla_put = pricer.price_option(82.50, 0.5, 'put', method='monte_carlo')

    barriers = [75, 70, 65, 60]
    print(f"  Vanilla Put (K=$82.50):   ${vanilla_put.price:.4f}")
    print(f"\n  Down-and-Out Put (K=$82.50, T=6M):")
    print(f"  {'Barrier':>10}  {'Price':>8}  {'Breach Prob':>12}  {'Discount':>10}")
    print("  " + "-" * 46)
    for B in barriers:
        res = pricer.price_barrier(82.50, 0.5, B, 'down-and-out', 'put')
        discount_pct = (1 - res.price / vanilla_put.price) * 100
        breach = res.metadata['breach_probability']
        print(f"  ${B:>8.1f}  ${res.price:>6.4f}  {breach:>11.1%}  "
              f"{discount_pct:>9.1f}%")


    # ── 7. SCHWARTZ MEAN-REVERSION ───────────────────────────
    separator("7. SCHWARTZ MODEL vs GBM (Mean-Reversion)")
    print("\n  Oil prices mean-revert — Schwartz (1997) captures this.")
    print("  Difference is larger for longer maturities.\n")

    print(f"  {'Maturity':>10}  {'GBM MC':>10}  {'Schwartz MC':>12}  {'Diff':>8}")
    print("  " + "-" * 46)
    for T in [0.25, 0.5, 1.0, 1.5, 2.0]:
        gbm = pricer.price_option(82.50, T, 'call', method='monte_carlo')
        sch = pricer.price_option(82.50, T, 'call', method='schwartz_mc')
        diff = sch.price - gbm.price
        print(f"  {T:>9.2f}Y  ${gbm.price:>8.4f}  ${sch.price:>10.4f}  "
              f"${diff:>+7.4f}")


    # ── 8. VOLATILITY SURFACE ─────────────────────────────────
    separator("8. IMPLIED VOLATILITY SURFACE")
    print("\n  Synthetic vol surface (oil skew: OTM puts expensive)\n")

    surface = pricer.vol_surface.synthetic_surface(atm_vol=0.32)
    maturities = sorted(surface.keys())

    # Print subset of surface
    sample_strikes = [57.75, 66.0, 74.25, 82.5, 90.75, 99.0]
    print(f"  {'Strike':>8}", end="")
    for T in maturities:
        label = f"{int(T*12)}M" if T < 1 else f"{T}Y"
        print(f"  {label:>8}", end="")
    print()
    print("  " + "-" * (10 + 10 * len(maturities)))

    for K in sample_strikes:
        K_label = "ATM" if K == 82.5 else ""
        print(f"  ${K:>5.2f} {K_label:>3}", end="")
        for T in maturities:
            K_closest = min(surface[T].keys(), key=lambda x: abs(x - K))
            iv = surface[T][K_closest]
            if iv is not None:
                print(f"  {iv*100:>7.1f}%", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        print()

    print("\n  (Downward skew = put options more expensive, typical for oil)")


    # ── 9. IMPLIED VOL INVERSION ─────────────────────────────
    separator("9. IMPLIED VOLATILITY — INVERSION EXAMPLE")
    print("\n  Given a market price, back out the implied vol\n")

    test_cases = [
        (82.50, 0.5, 'call', 8.50),
        (80.00, 0.5, 'put',  6.20),
        (90.00, 0.5, 'call', 4.15),
        (75.00, 0.5, 'put',  7.80),
    ]

    print(f"  {'Strike':>8}  {'Type':>6}  {'Mkt Price':>10}  {'Impl. Vol':>10}")
    print("  " + "-" * 42)
    for K, T, opt, mkt_price in test_cases:
        iv = pricer.implied_vol(mkt_price, K, T, opt)
        if iv:
            print(f"  ${K:>6.2f}  {opt:>6}  ${mkt_price:>8.2f}  {iv*100:>9.2f}%")

    separator()
    print("\n  PRICER COMPLETE\n")


if __name__ == "__main__":
    main()
