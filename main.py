# Read in input file
def ReadInputFile(filename):
"""Reads input file containing problem data.

First line of the input file contains the number of variables. All of these
are assumed to be binary (0-1) variables.

The second line contains the coefficients of each variable in the single
knapsack constraint, delimited by commas.

The third line contains the right hand side value for the knapsack constraint.
This value is assumed to be integer.
TODO(jwd): check if this requirement is necessary.

Inputs:
    filename: A string, the filename of the input file to read.

Returns:
    A list, the coefficients for the knapsack constraint.
    A number, the value for the right hand side in the knapsack constraint.

"""
    pass

# Order set of coefficients

# Find min covers of constraint

# Filter out all non strong covers

# Generate constraint from each strong cover

# Check constraints for facet defining condition

# Generate output file
