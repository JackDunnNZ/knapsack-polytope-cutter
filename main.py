
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

    f = open(filename, 'r')

    # Read number of coefficients
    n = int(f.readline().rstrip('\n'))

    # Read in n coefficients
    A = [int(a) for a in f.readline().rstrip('\n').split(',')[0:n]]

    # Read RHS
    b = int(f.readline().rstrip('\n'))

    f.close()
    return A, b

# Order set of coefficients
def ConstructOrderedSet(A, b):
    # Placeholder
    N = set(range(1, len(A) + 1))
    return N

# Find min covers of constraint
def GenerateMinimalCovers(A, b):
    pass

# Filter out all non strong covers

# Generate constraint from each strong cover

# Check constraints for facet defining condition

# Generate output file

# Main routine (to be moved)
A, b = ReadInputFile('example_problem.dat')
N = ConstructOrderedSet(A, b)
