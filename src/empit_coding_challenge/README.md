# EMPIT Coding Interview Challenge

Execute this python module (empit_coding_challenge) like this:

    python -m empit_coding_challenge

## Tasks
1. Take a look at the Polynom class in `polynom.py`. It represents a polynom based on a list of coefficients.
    1. Fix the add method
    2. Add a subtract and multiply method, so we can do p-q and p*q for polynoms p and q easily

2. Create 3 different solvers in `solvers.py` for solving the integral of a polynom p in an interval \[a,b\] 
    1. An analytic solver, that calculates the coefficients of the intagral like you would do by hand and evaluates it for the interval
    2. A numeric solver, that calculates the integral by looking at n equidistant values of the polynom p
    3. A stochastic solver, that calculates the integral by using a monte carlo simulation with n samples of the polynom p

3. Calculate the integral a polynom p and an interval \[a,b\] given via command line arguments in `main.py`
    1. Create a polynom and interval out of the arguments given via the cl
    2. Solve the integral using your different solvers
    3. Show the solution and execution time of each solver

4. Package your code as a wheel, that we can install and then run the command above.

4. Think about how you could speed up the numeric solvers.
