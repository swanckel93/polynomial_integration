#!/bin/bash

# Polynomial integration script
# Integrates the polynomial with coefficients "1 0 3 5 7 9 9 -200 -40 -25 9"
# over the interval [-128, 128] 
# WARNING: might take a while
# Execute me from root directory.

poly integrate --tolerance 1e-11 --timeout 20 --start-n 10 --batch-size 1024 -- "1 0 3 5 7 9 9 -200 -40 -25 9" -128 128