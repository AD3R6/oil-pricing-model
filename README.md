# Oil Derivatives Pricer

A Python pricing engine for oil derivatives, built for economists and finance practitioners. Prices futures, vanilla options, and exotic derivatives on crude oil using industry-standard numerical methods.

---

## Models Implemented

### Futures & Forwards
Cost-of-carry model with explicit convenience yield and storage costs:
```
F = S × exp((r + storage - convenience_yield) × T)
```

### Black-76 (European Options)
The industry standard for pricing European options on futures. Avoids the need to model convenience yield explicitly by using the futures price as the underlying. Includes full analytical Greeks (Δ, Γ, ν, Θ, ρ).

### Binomial Tree — CRR (European & American Options)
Cox-Ross-Rubinstein lattice model. Supports American early exercise, which matters for deep in-the-money oil puts. Greeks computed via finite differences.

### Monte Carlo — GBM
Geometric Brownian Motion simulation (50,000 paths). Supports:
- European vanilla options
- **Asian options** (arithmetic & geometric average) — widely used in physical oil contracts with Platts/MOPS average pricing
- **Barrier options** (down-and-out, up-and-out, knock-in variants) — common for reducing hedging costs

### Schwartz Mean-Reversion Model
Schwartz (1997) one-factor model. Oil prices mean-revert to a long-run level, unlike equities. This makes a material difference for maturities beyond 6 months and produces lower long-dated option prices than GBM.

### Implied Volatility Surface
Inverts Black-76 to back out implied volatility from market prices. Includes a synthetic surface generator that replicates realistic oil skew (OTM puts more expensive than OTM calls, short-end vol elevated).

---

## Installation

**Requirements:** Python 3.8+

Install dependencies:
```bash
pip3 install numpy scipy
```

No other libraries needed.

---

## Quickstart

### Terminal
```bash
cd /path/to/Pricer
python3 demo.py
```

### Jupyter Notebook
```python
import sys
sys.path.append("/path/to/Pricer")

from pricer import OilDerivativesPricer, OilMarketData

market = OilMarketData(
    spot_price=82.50,
    risk_free_rate=0.0525,
    convenience_yield=0.035,
    storage_cost=0.015,
    volatility=0.32
)

p = OilDerivativesPricer(market)
```

---

## Usage Examples

### Futures Curve
```python
curve = p.futures_curve(maturity=1.0)
print(curve["futures_price"])       # front price
print(curve["term_structure"])      # 1M to 2Y term structure
```

### European Option (Black-76)
```python
result = p.price_option(strike=85, maturity=0.5, option_type='call', method='black76')
print(result.price)     # option price
print(result.greeks)    # {'delta': ..., 'gamma': ..., 'vega': ..., 'theta': ..., 'rho': ...}
```

### American Option (Binomial Tree)
```python
result = p.price_option(strike=80, maturity=1.0, option_type='put',
                        exercise='american', method='binomial')
print(result.price)
```

### Asian Option (Monte Carlo)
```python
result = p.price_asian(strike=82.50, maturity=0.5,
                       option_type='call', average_type='arithmetic')
print(result.price)
print(result.metadata['confidence_interval_95'])
```

### Barrier Option
```python
result = p.price_barrier(strike=82.50, maturity=0.5,
                         barrier=70.0, barrier_type='down-and-out',
                         option_type='put')
print(result.price)
print(result.metadata['breach_probability'])
```

### Schwartz Mean-Reversion
```python
result = p.price_option(strike=82.50, maturity=2.0,
                        option_type='call', method='schwartz_mc')
print(result.price)
```

### Implied Volatility
```python
iv = p.implied_vol(market_price=8.50, strike=82.50,
                   maturity=0.5, option_type='call')
print(f"Implied vol: {iv*100:.2f}%")
```

### Compare All Methods
```python
summary = p.summary(strike=82.50, maturity=0.5, option_type='call')
for method, price in summary["prices"].items():
    print(f"{method}: ${price:.4f}")
print(f"Early exercise premium: ${summary['early_exercise_premium']:.4f}")
```

---

## Market Data Inputs

| Parameter | Description | Typical WTI Value |
|---|---|---|
| `spot_price` | Current spot price ($/bbl) | ~80 |
| `risk_free_rate` | Annualized risk-free rate | 0.05 |
| `convenience_yield` | Net convenience yield | 0.03 |
| `storage_cost` | Annualized storage cost | 0.015 |
| `volatility` | Annualized implied volatility | 0.25–0.40 |

Cost of carry is derived automatically:
```
b = risk_free_rate + storage_cost - convenience_yield
```

---

## File Structure

```
Pricer/
├── pricer.py      # Pricing engine (all models)
├── demo.py        # Full walkthrough with WTI parameters
└── README.md      # This file
```

---

## Key Economic Concepts

**Convenience yield** — the implicit benefit of holding physical crude (e.g. feedstock availability, supply disruption protection). Higher convenience yield flattens or inverts the futures curve.

**Mean reversion** — oil prices tend to revert toward long-run marginal cost of production. The Schwartz model captures this with a mean-reversion speed parameter `kappa`. Ignoring mean reversion (pure GBM) overstates long-dated option prices.

**Oil skew** — implied volatility is higher for OTM puts than OTM calls in crude markets, reflecting asymmetric downside risk (demand shocks, OPEC supply increases). The vol surface generator replicates this shape.

**Asian pricing** — most physical oil contracts settle on the monthly average of daily prices (e.g. Platts Dated Brent). Asian options are therefore more economically relevant than vanilla Europeans for hedging physical exposure.

---

## References

- Black, F. (1976). *The Pricing of Commodity Contracts*. Journal of Financial Economics.
- Cox, J., Ross, S., Rubinstein, M. (1979). *Option Pricing: A Simplified Approach*. Journal of Financial Economics.
- Schwartz, E. (1997). *The Stochastic Behavior of Commodity Prices*. Journal of Finance.
