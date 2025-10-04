"""
Legacy train.py - Re-exports from permuted_puzzle.train_utils for backward compatibility.
The actual implementations have been moved to permuted_puzzle/train_utils.py as part of the library.
"""

from permuted_puzzle.train_utils import train_model, evaluate_model

__all__ = ['train_model', 'evaluate_model']
