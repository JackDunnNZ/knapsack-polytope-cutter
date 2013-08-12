from itertools import chain, combinations
from collections import defaultdict

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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

def SumCoeffsOverSet(summing_set, A):
    return sum(A[i - 1] for i in summing_set)

# Find min covers of constraint
def GenerateMinimalCovers(N, A, b):
    sets = []
    set_map = defaultdict(list)
    for subset in powerset(N):
        if SumCoeffsOverSet(subset, A) > b:
            print subset, SumCoeffsOverSet(subset, A)
            sets.append(subset)
            set_map[len(subset)].append(subset)

    # Filter out all non strong covers
    for set_length, subsets in set_map.iteritems():
        l = len(subsets)
        for i, subset in enumerate(reversed(subsets)):
            for j in range(i, l):
                print l, i, j
                if subset < subsets[l - 1 - j]:
                    subsets.remove(subset)
        print subsets

# Generate constraint from each strong cover

# Check constraints for facet defining condition

# Generate output file

# Main routine (to be moved)
A, b = ReadInputFile('example_problem.dat')
N = ConstructOrderedSet(A, b)
GenerateMinimalCovers(N, A, b)
