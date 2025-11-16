"""
Kirby's Dream Land RL Training Pipeline
"""
from .callbacks import StatsCallback
from .ppo_lambda_discrepancy import CnnLstmPolicyLD, RecurrentPPOLD

__version__ = "0.1.0"
__all__ = ["StatsCallback", "CnnLstmPolicyLD", "RecurrentPPOLD", "__version__"]
