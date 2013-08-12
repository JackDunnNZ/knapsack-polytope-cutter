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

def SumCoeffsOverIndexList(summing_list, A):
    return sum(A[i] * summing_list[i] for i in range(len(summing_list)))

def ConvertIndexListToSet(index_list):
    return set(i + 1 for i, j in enumerate(index_list) if j == 1)

def ExtendSet(index_set):
    min_elem = min(index_set)
    return index_set.union(set(range(1, min_elem)))

# Find min covers of constraint
def GenerateMinimalCovers(N, A, b):
    sets = []
    set_map = defaultdict(list)

    n = len(N)
    s = [0 for _ in range(n)]

    k = 0
    while k < n:
        print s, k
        v = SumCoeffsOverIndexList(s, A)
        s[k] += 1
        if v + A[k] > b:
            subset = ConvertIndexListToSet(s)
            set_map[len(subset)].append(subset)
            s[k] = 0
        k += 1
        if k == n:
            if v == 0:
                break
            k -= 1
            s[k] = 0
            while s[k] != 1:
                s[k] = 0
                k -= 1
            s[k] = 0
            k += 1

    # Filter out all non strong covers
    for subsets in set_map.itervalues():
        extended_subsets = [ExtendSet(ss) for ss in subsets]
        l = len(subsets) - 1
        for i in range(len(subsets)):
            for j in range(len(subsets)):
                if extended_subsets[l - i] < extended_subsets[l - j]:
                    subsets.pop(l-i)
                    break

        sets += subsets

    return sets


# Generate constraint from each strong cover

# Check constraints for facet defining condition

# Generate output file

# Main routine (to be moved)
A, b = ReadInputFile('example_problem.dat')
N = ConstructOrderedSet(A, b)
sets = GenerateMinimalCovers(N, A, b)
