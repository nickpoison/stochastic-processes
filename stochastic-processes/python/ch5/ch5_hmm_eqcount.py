"""
Chapter 5 - Example 1.7 / Ch.5.6: a 2-state Poisson HMM on EQcount.

R:
  library(astsa)
  fm <- HmmFit(EQcount, m=2)  # default is Poisson
  fm$Pmatrix
  fm$lambda

This reproduces the book's printed numbers almost exactly (validated
when this was built: lambda [15.4207, 26.0181] vs. book's [15.4207,
26.0181]; Pmatrix matching to 4+ decimal places).

Run: python ch5/ch5_hmm_eqcount.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stocproc_tools import load
from stocproc_tools.hmm import hmm_fit, viterbi

eq = load("EQcount").values

fm = hmm_fit(eq, m=2, family="pois", seed=42, se="none")

print("Transition matrix (Pmatrix):")
print(np.round(fm["Pmatrix"], 4))
print()
print("Lambda estimates:")
print(np.round(fm["lambda"], 4))
print()
print("Book's printed values, for comparison:")
print("  Pmatrix: [[0.9284, 0.0716], [0.1190, 0.8810]]")
print("  lambda:  [15.4207, 26.0181]")

states = viterbi(fm)
print()
print("Decoded regime for the last 10 years:")
print(states[-10:])
