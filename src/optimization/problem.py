"""Multi-objective formulation of the microgrid energy market for jMetal.

Decision variables (FloatProblem):  [q_solar, q_wind, p_solar, p_wind]
Objectives (3):  maximize profit_solar, maximize profit_wind, minimize buyer_cost
Constraint (1):  cover the demand (no shortfall)

jMetal minimizes every objective, so the two profit objectives are stored
**negated**; use :func:`decode_objectives` to recover human-readable values.
"""

from __future__ import annotations

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from src.optimization.market import MarketConfig, clear_market

OBJ_LABELS = ["Profit Solar (AS)", "Profit Wind (AE)", "Buyer Cost (AC)"]


class EnergyMarketProblem(FloatProblem):
    """Two competing renewable sellers and one consumer."""

    def __init__(self, cfg: MarketConfig):
        super().__init__()
        self.cfg = cfg

        # Variable bounds: quantities limited by forecasted generation, prices in
        # the configured price band.
        self.lower_bound = [0.0, 0.0, cfg.price_min, cfg.price_min]
        self.upper_bound = [cfg.gen_solar, cfg.gen_wind, cfg.price_max, cfg.price_max]

        self.directions = [self.MAXIMIZE, self.MAXIMIZE, self.MINIMIZE]
        self.labels = OBJ_LABELS

    def number_of_variables(self) -> int:
        return 4

    def number_of_objectives(self) -> int:
        return 3

    def number_of_constraints(self) -> int:
        return 1

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        q_solar, q_wind, p_solar, p_wind = solution.variables
        out = clear_market(q_solar, q_wind, p_solar, p_wind, self.cfg)

        # jMetal minimizes: negate the objectives we want to maximize.
        solution.objectives[0] = -out.profit_solar
        solution.objectives[1] = -out.profit_wind
        solution.objectives[2] = out.buyer_cost

        # Constraint convention in jMetalPy: >= 0 is feasible, < 0 is a violation.
        solution.constraints[0] = -out.shortfall
        return solution

    def name(self) -> str:
        return "EnergyMarketProblem"


def decode_objectives(objectives) -> tuple[float, float, float]:
    """Turn stored (minimization) objectives back into human-readable values:
    (profit_solar, profit_wind, buyer_cost)."""
    return (-objectives[0], -objectives[1], objectives[2])


OBJ_LABELS_BATTERY = ["Profit Solar (AS)", "Profit Wind (AE)",
                      "Buyer Cost (AC)", "Battery Profit"]


class EnergyMarketBatteryProblem(FloatProblem):
    """Many-objective (4) variant with a storage **arbitrageur** (M1, opt side).

    Adds two decision variables — the quantity the battery buys (``q_batt``) and a
    resale **markup** ``f ∈ [1, 1.2]`` (treated like the sellers' price variables) —
    and a 4th objective: maximize the battery's profit ``= bought · price · (f−1)``.
    This makes the problem genuinely **many-objective** (motivates NSGA-III). The
    markup is a PROXY for the temporal spread (the battery "sells later at +0–20%");
    the real arbitrage is validated temporally in ``src/mas/battery.py``.
    """

    def __init__(self, cfg: MarketConfig, markup_max: float = 1.2):
        super().__init__()
        self.cfg = cfg
        self.markup_max = markup_max
        cap = cfg.gen_solar + cfg.gen_wind
        #               q_solar      q_wind      p_solar        p_wind       q_batt markup
        self.lower_bound = [0.0, 0.0, cfg.price_min, cfg.price_min, 0.0, 1.0]
        self.upper_bound = [cfg.gen_solar, cfg.gen_wind, cfg.price_max, cfg.price_max,
                            cap, markup_max]
        self.directions = [self.MAXIMIZE, self.MAXIMIZE, self.MINIMIZE, self.MAXIMIZE]
        self.labels = OBJ_LABELS_BATTERY

    def number_of_variables(self) -> int:
        return 6

    def number_of_objectives(self) -> int:
        return 4

    def number_of_constraints(self) -> int:
        return 1

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        q_solar, q_wind, p_solar, p_wind, q_batt, markup = solution.variables
        out = clear_market(q_solar, q_wind, p_solar, p_wind, self.cfg,
                           q_batt=q_batt, markup=markup)
        solution.objectives[0] = -out.profit_solar
        solution.objectives[1] = -out.profit_wind
        solution.objectives[2] = out.buyer_cost
        solution.objectives[3] = -out.battery_profit
        solution.constraints[0] = -out.shortfall
        return solution

    def name(self) -> str:
        return "EnergyMarketBatteryProblem"


def decode_objectives_battery(objectives) -> tuple[float, float, float, float]:
    """(profit_solar, profit_wind, buyer_cost, battery_profit) from stored objectives."""
    return (-objectives[0], -objectives[1], objectives[2], -objectives[3])
