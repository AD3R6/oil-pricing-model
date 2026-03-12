"""
Oil Derivatives Pricer
======================
Pricing engine for oil derivatives using numerical methods:
- European & American Options (Black-76, Binomial Tree, Monte Carlo)
- Asian Options (Monte Carlo)
- Futures & Forwards (Cost-of-Carry with convenience yield)
- Spread Options (Kirk's Approximation + Monte Carlo)

Oil-specific features:
- Schwartz mean-reversion model for spot prices
- Convenience yield term structure
- Seasonal volatility adjustments
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class OilMarketData:
    """Current market inputs for oil derivatives"""
    spot_price: float          # $/barrel (e.g., 80.0)
    risk_free_rate: float      # annualized (e.g., 0.05)
    convenience_yield: float   # net convenience yield (e.g., 0.03)
    storage_cost: float        # annualized (e.g., 0.02)
    volatility: float          # annualized vol (e.g., 0.30)
    
    @property
    def cost_of_carry(self):
        """b = r + storage - convenience_yield"""
        return self.risk_free_rate + self.storage_cost - self.convenience_yield


@dataclass
class OptionParams:
    """Option contract specification"""
    strike: float
    maturity: float            # years to expiry
    option_type: str           # 'call' or 'put'
    exercise: str = 'european' # 'european' or 'american'


@dataclass
class PricingResult:
    """Output of pricing calculation"""
    price: float
    method: str
    greeks: dict
    metadata: dict = None


# ─────────────────────────────────────────────
# FUTURES & FORWARDS
# ─────────────────────────────────────────────

class FuturesPricer:
    """
    Oil futures pricing via cost-of-carry model.
    F = S * exp((r + storage - convenience_yield) * T)
    """

    def price(self, market: OilMarketData, maturity: float) -> dict:
        S = market.spot_price
        b = market.cost_of_carry
        T = maturity

        futures_price = S * np.exp(b * T)

        # Term structure: prices at multiple maturities
        maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
        term_structure = S * np.exp(b * maturities)

        return {
            "futures_price": round(futures_price, 4),
            "spot_price": S,
            "maturity": T,
            "cost_of_carry": round(b, 4),
            "term_structure": {
                f"{int(m*12)}M" if m < 1 else f"{m}Y": round(p, 2)
                for m, p in zip(maturities, term_structure)
            }
        }


# ─────────────────────────────────────────────
# BLACK-76 MODEL (European Options on Futures)
# ─────────────────────────────────────────────

class Black76Pricer:
    """
    Black (1976) model — the industry standard for
    pricing European options on oil futures.
    
    Uses futures price F as the underlying, avoids
    need to model convenience yield explicitly.
    """

    def _d1_d2(self, F, K, sigma, T):
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def price(self, market: OilMarketData, params: OptionParams) -> PricingResult:
        S, r, b, sigma = (market.spot_price, market.risk_free_rate,
                          market.cost_of_carry, market.volatility)
        K, T = params.strike, params.maturity
        opt = params.option_type.lower()

        # Futures price
        F = S * np.exp(b * T)
        discount = np.exp(-r * T)

        d1, d2 = self._d1_d2(F, K, sigma, T)

        if opt == 'call':
            price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

        greeks = self._greeks(F, K, sigma, T, r, d1, d2, opt, discount)

        return PricingResult(
            price=round(price, 4),
            method="Black-76",
            greeks=greeks,
            metadata={"futures_price": round(F, 4), "d1": round(d1, 4), "d2": round(d2, 4)}
        )

    def implied_vol(self, market_price: float, market: OilMarketData,
                    params: OptionParams) -> float:
        """Solve for implied volatility via Brent's method"""
        S, r, b = market.spot_price, market.risk_free_rate, market.cost_of_carry
        K, T, opt = params.strike, params.maturity, params.option_type

        F = S * np.exp(b * T)
        discount = np.exp(-r * T)

        def objective(sigma):
            d1, d2 = self._d1_d2(F, K, sigma, T)
            if opt == 'call':
                p = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
            else:
                p = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
            return p - market_price

        try:
            return round(brentq(objective, 1e-6, 5.0), 6)
        except ValueError:
            return None

    def _greeks(self, F, K, sigma, T, r, d1, d2, opt, discount):
        sqrt_T = np.sqrt(T)
        pdf_d1 = norm.pdf(d1)

        vega = discount * F * pdf_d1 * sqrt_T / 100  # per 1% vol move

        if opt == 'call':
            delta = discount * norm.cdf(d1)
            theta = (-discount * F * pdf_d1 * sigma / (2 * sqrt_T)
                     - r * discount * (F * norm.cdf(d1) - K * norm.cdf(d2))) / 365
            rho = -T * discount * (F * norm.cdf(d1) - K * norm.cdf(d2)) / 100
        else:
            delta = -discount * norm.cdf(-d1)
            theta = (-discount * F * pdf_d1 * sigma / (2 * sqrt_T)
                     - r * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))) / 365
            rho = -T * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1)) / 100

        gamma = discount * pdf_d1 / (F * sigma * sqrt_T)

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "vega":  round(vega, 4),
            "theta": round(theta, 4),
            "rho":   round(rho, 4)
        }


# ─────────────────────────────────────────────
# BINOMIAL TREE (American Options)
# ─────────────────────────────────────────────

class BinomialTreePricer:
    """
    Cox-Ross-Rubinstein binomial tree.
    Supports American early exercise — critical for
    in-the-money oil puts and deep ITM calls.
    """

    def __init__(self, steps: int = 500):
        self.steps = steps

    def price(self, market: OilMarketData, params: OptionParams) -> PricingResult:
        S, r, b, sigma = (market.spot_price, market.risk_free_rate,
                          market.cost_of_carry, market.volatility)
        K, T, opt, exercise = (params.strike, params.maturity,
                                params.option_type.lower(), params.exercise.lower())

        N = self.steps
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(b * dt) - d) / (u - d)  # risk-neutral prob
        discount = np.exp(-r * dt)

        # Terminal stock prices
        j = np.arange(N + 1)
        ST = S * (u ** (N - j)) * (d ** j)

        # Terminal payoffs
        if opt == 'call':
            V = np.maximum(ST - K, 0)
        else:
            V = np.maximum(K - ST, 0)

        # Backward induction
        for i in range(N - 1, -1, -1):
            ST = S * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
            V = discount * (p * V[:i+1] + (1 - p) * V[1:i+2])

            if exercise == 'american':
                if opt == 'call':
                    V = np.maximum(V, ST - K)
                else:
                    V = np.maximum(V, K - ST)

        price = V[0]

        # Greeks via finite differences
        greeks = self._greeks_fd(market, params, price)

        return PricingResult(
            price=round(price, 4),
            method=f"Binomial Tree (N={N}, {exercise.capitalize()})",
            greeks=greeks,
            metadata={"steps": N, "up_factor": round(u, 4), "risk_neutral_prob": round(p, 4)}
        )

    def _greeks_fd(self, market, params, base_price):
        """Finite difference Greeks using a simpler tree (no recursive Greeks)"""
        eps_S = market.spot_price * 0.01
        eps_v = 0.01

        def simple_price(mkt, prm):
            """Price without computing Greeks (breaks recursion)"""
            S, r, b, sigma = mkt.spot_price, mkt.risk_free_rate, mkt.cost_of_carry, mkt.volatility
            K, T, opt, exercise = prm.strike, prm.maturity, prm.option_type.lower(), prm.exercise.lower()
            N = 150
            dt = T / N
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(b * dt) - d) / (u - d)
            disc = np.exp(-r * dt)
            j = np.arange(N + 1)
            ST = S * (u ** (N - j)) * (d ** j)
            V = np.maximum(ST - K, 0) if opt == 'call' else np.maximum(K - ST, 0)
            for i in range(N - 1, -1, -1):
                ST_i = S * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
                V = disc * (p * V[:i+1] + (1 - p) * V[1:i+2])
                if exercise == 'american':
                    V = np.maximum(V, ST_i - K) if opt == 'call' else np.maximum(V, K - ST_i)
            return V[0]

        # Delta & Gamma
        m_up = OilMarketData(market.spot_price + eps_S, market.risk_free_rate,
                              market.convenience_yield, market.storage_cost, market.volatility)
        m_dn = OilMarketData(market.spot_price - eps_S, market.risk_free_rate,
                              market.convenience_yield, market.storage_cost, market.volatility)
        p_up = simple_price(m_up, params)
        p_dn = simple_price(m_dn, params)

        delta = (p_up - p_dn) / (2 * eps_S)
        gamma = (p_up - 2 * base_price + p_dn) / (eps_S ** 2)

        # Vega
        m_vup = OilMarketData(market.spot_price, market.risk_free_rate,
                               market.convenience_yield, market.storage_cost,
                               market.volatility + eps_v)
        p_vup = simple_price(m_vup, params)
        vega = (p_vup - base_price) / 100

        # Theta (1-day)
        p_t = OptionParams(params.strike, max(params.maturity - 1/365, 1e-6),
                           params.option_type, params.exercise)
        p_theta = simple_price(market, p_t)
        theta = p_theta - base_price

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "vega":  round(vega, 4),
            "theta": round(theta, 4)
        }


# ─────────────────────────────────────────────
# MONTE CARLO PRICER
# ─────────────────────────────────────────────

class MonteCarloPricer:
    """
    Monte Carlo simulation for exotic oil derivatives.
    
    Supports:
    - European vanilla options
    - Asian options (arithmetic & geometric average)
    - Lookback options
    - Barrier options
    
    Uses Geometric Brownian Motion with optional
    mean-reversion (Schwartz model for oil).
    """

    def __init__(self, n_paths: int = 50_000, n_steps: int = 252, seed: int = 42):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def _simulate_gbm(self, market: OilMarketData, T: float) -> np.ndarray:
        """Simulate GBM paths: shape (n_paths, n_steps+1)"""
        rng = np.random.default_rng(self.seed)
        S0, b, sigma = market.spot_price, market.cost_of_carry, market.volatility
        dt = T / self.n_steps
        N, M = self.n_paths, self.n_steps

        Z = rng.standard_normal((N, M))
        log_returns = (b - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(log_returns, axis=1)
        paths = S0 * np.exp(np.hstack([np.zeros((N, 1)), log_paths]))
        return paths

    def _simulate_schwartz(self, market: OilMarketData, T: float,
                            kappa: float = 0.5, theta_s: float = None) -> np.ndarray:
        """
        Schwartz (1997) mean-reversion model.
        dln(S) = kappa*(theta - ln(S))*dt + sigma*dW
        Better captures oil price mean-reversion.
        """
        rng = np.random.default_rng(self.seed)
        S0, sigma = market.spot_price, market.volatility
        if theta_s is None:
            theta_s = np.log(S0)  # long-run mean = current spot
        dt = T / self.n_steps
        N, M = self.n_paths, self.n_steps

        log_S = np.zeros((N, M + 1))
        log_S[:, 0] = np.log(S0)
        Z = rng.standard_normal((N, M))

        for t in range(M):
            log_S[:, t+1] = (log_S[:, t]
                             + kappa * (theta_s - log_S[:, t]) * dt
                             + sigma * np.sqrt(dt) * Z[:, t])
        return np.exp(log_S)

    def price_european(self, market: OilMarketData, params: OptionParams,
                       use_schwartz: bool = False) -> PricingResult:
        r, K, T, opt = (market.risk_free_rate, params.strike,
                        params.maturity, params.option_type.lower())

        paths = (self._simulate_schwartz(market, T) if use_schwartz
                 else self._simulate_gbm(market, T))
        ST = paths[:, -1]

        if opt == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        se = discount * np.std(payoffs) / np.sqrt(self.n_paths)

        model = "Schwartz Mean-Reversion MC" if use_schwartz else "GBM Monte Carlo"
        return PricingResult(
            price=round(price, 4),
            method=model,
            greeks={},
            metadata={
                "std_error": round(se, 4),
                "confidence_interval_95": (round(price - 1.96*se, 4), round(price + 1.96*se, 4)),
                "n_paths": self.n_paths
            }
        )

    def price_asian(self, market: OilMarketData, params: OptionParams,
                    average_type: str = 'arithmetic') -> PricingResult:
        """Asian option: payoff based on average price over life"""
        r, K, T, opt = (market.risk_free_rate, params.strike,
                        params.maturity, params.option_type.lower())

        paths = self._simulate_gbm(market, T)

        if average_type == 'arithmetic':
            avg = np.mean(paths[:, 1:], axis=1)
        else:
            avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

        if opt == 'call':
            payoffs = np.maximum(avg - K, 0)
        else:
            payoffs = np.maximum(K - avg, 0)

        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        se = discount * np.std(payoffs) / np.sqrt(self.n_paths)

        return PricingResult(
            price=round(price, 4),
            method=f"Asian Option MC ({average_type} avg)",
            greeks={},
            metadata={
                "std_error": round(se, 4),
                "confidence_interval_95": (round(price - 1.96*se, 4), round(price + 1.96*se, 4)),
                "average_type": average_type
            }
        )

    def price_barrier(self, market: OilMarketData, params: OptionParams,
                      barrier: float, barrier_type: str = 'down-and-out') -> PricingResult:
        """
        Barrier option — common in oil markets for
        capping hedging costs (e.g., down-and-out puts).
        barrier_type: 'down-and-out', 'up-and-out', 'down-and-in', 'up-and-in'
        """
        r, K, T, opt = (market.risk_free_rate, params.strike,
                        params.maturity, params.option_type.lower())

        paths = self._simulate_gbm(market, T)
        ST = paths[:, -1]

        btype = barrier_type.lower()
        if 'down' in btype:
            breached = np.any(paths <= barrier, axis=1)
        else:
            breached = np.any(paths >= barrier, axis=1)

        if opt == 'call':
            vanilla_payoff = np.maximum(ST - K, 0)
        else:
            vanilla_payoff = np.maximum(K - ST, 0)

        if 'out' in btype:
            payoffs = np.where(breached, 0, vanilla_payoff)
        else:  # 'in'
            payoffs = np.where(breached, vanilla_payoff, 0)

        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        se = discount * np.std(payoffs) / np.sqrt(self.n_paths)

        return PricingResult(
            price=round(price, 4),
            method=f"Barrier Option MC ({barrier_type})",
            greeks={},
            metadata={
                "barrier": barrier,
                "barrier_type": barrier_type,
                "breach_probability": round(np.mean(breached), 4),
                "std_error": round(se, 4)
            }
        )


# ─────────────────────────────────────────────
# VOLATILITY SURFACE
# ─────────────────────────────────────────────

class VolatilitySurface:
    """
    Build implied volatility surface from option prices.
    Uses Black-76 inversion across strikes and maturities.
    """

    def __init__(self, market: OilMarketData):
        self.market = market
        self.b76 = Black76Pricer()

    def build(self, strikes: np.ndarray, maturities: np.ndarray,
              option_prices: np.ndarray) -> dict:
        """
        Invert Black-76 to get implied vols.
        option_prices: shape (len(maturities), len(strikes))
        """
        surface = {}
        for i, T in enumerate(maturities):
            surface[T] = {}
            for j, K in enumerate(strikes):
                params = OptionParams(K, T, 'call')
                iv = self.b76.implied_vol(option_prices[i, j], self.market, params)
                surface[T][K] = iv
        return surface

    def synthetic_surface(self, atm_vol: float = None) -> dict:
        """
        Generate a realistic synthetic vol surface for oil
        using typical term structure and skew shapes.
        """
        if atm_vol is None:
            atm_vol = self.market.volatility

        S = self.market.spot_price
        strikes = np.array([0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]) * S
        maturities = np.array([1/12, 3/12, 6/12, 1.0, 2.0])

        surface = {}
        for T in maturities:
            surface[T] = {}
            for K in strikes:
                moneyness = np.log(K / S) / (atm_vol * np.sqrt(T))
                # Oil skew: puts more expensive (downside fear)
                skew = -0.05 * moneyness
                # Term structure: short-term vol higher
                term_adj = atm_vol * (1 + 0.1 * np.exp(-2 * T))
                smile = 0.02 * moneyness**2
                iv = term_adj + skew + smile
                surface[T][round(K, 2)] = round(max(iv, 0.05), 4)

        return surface


# ─────────────────────────────────────────────
# CONVENIENCE INTERFACE
# ─────────────────────────────────────────────

class OilDerivativesPricer:
    """
    Master pricer interface.
    Wraps all models for easy use.
    """

    def __init__(self, market: OilMarketData):
        self.market = market
        self.futures = FuturesPricer()
        self.black76 = Black76Pricer()
        self.tree = BinomialTreePricer()
        self.mc = MonteCarloPricer()
        self.vol_surface = VolatilitySurface(market)

    def price_option(self, strike, maturity, option_type='call',
                     exercise='european', method='black76') -> PricingResult:
        params = OptionParams(strike, maturity, option_type, exercise)
        if method == 'black76':
            return self.black76.price(self.market, params)
        elif method == 'binomial':
            return self.tree.price(self.market, params)
        elif method == 'monte_carlo':
            return self.mc.price_european(self.market, params)
        elif method == 'schwartz_mc':
            return self.mc.price_european(self.market, params, use_schwartz=True)
        else:
            raise ValueError(f"Unknown method: {method}")

    def price_asian(self, strike, maturity, option_type='call',
                    average_type='arithmetic') -> PricingResult:
        params = OptionParams(strike, maturity, option_type)
        return self.mc.price_asian(self.market, params, average_type)

    def price_barrier(self, strike, maturity, barrier, barrier_type='down-and-out',
                      option_type='put') -> PricingResult:
        params = OptionParams(strike, maturity, option_type)
        return self.mc.price_barrier(self.market, params, barrier, barrier_type)

    def futures_curve(self, maturity=1.0) -> dict:
        return self.futures.price(self.market, maturity)

    def implied_vol(self, market_price, strike, maturity, option_type='call') -> float:
        params = OptionParams(strike, maturity, option_type)
        return self.black76.implied_vol(market_price, self.market, params)

    def summary(self, strike, maturity, option_type='call') -> dict:
        """Compare all methods for a single option"""
        params = OptionParams(strike, maturity, option_type, 'european')
        b76 = self.black76.price(self.market, params)

        params_am = OptionParams(strike, maturity, option_type, 'american')
        tree_eu = BinomialTreePricer(300).price(self.market, params)
        tree_am = BinomialTreePricer(300).price(self.market, params_am)
        mc = self.mc.price_european(self.market, params)

        return {
            "option": f"{option_type.upper()} K={strike} T={maturity}Y",
            "spot": self.market.spot_price,
            "vol": self.market.volatility,
            "prices": {
                "Black-76 (European)": b76.price,
                "Binomial Tree (European)": tree_eu.price,
                "Binomial Tree (American)": tree_am.price,
                "Monte Carlo (GBM)": mc.price,
            },
            "greeks": b76.greeks,
            "early_exercise_premium": round(tree_am.price - tree_eu.price, 4)
        }
