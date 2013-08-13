import time
t_ = time.clock()

import argparse
from collections import defaultdict

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
    sort_map = sorted(range(len(A)), key=A.__getitem__, reverse=True)
    A.sort(reverse=True)
    # Placeholder
    N = set(range(1, len(A) + 1))
    return N, A, sort_map

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

def SetCoefficientsForIndexSet(index_set, pi, value):
    for i in index_set:
        pi[i - 1] = value
    return pi

def FirstNElemsOfSet(index_set, n):
    return set(sorted(list(index_set))[:n])

def SumFirstNCoeffsOfSet(index_set, A, n):
    reduced_set = FirstNElemsOfSet(index_set, n)
    return SumCoeffsOverSet(reduced_set, A)

def CheckValidityOfCut(S, A, h, Nh, b):
    reduced_S = S - FirstNElemsOfSet(S, h + 1)
    reduced_sum = SumCoeffsOverSet(reduced_S, A)
    for i in Nh:
        if reduced_sum + A[i - 1] > b:
            return False
    return True

def ReverseSortMap(A, sort_map):
    reversed_A = [0 for _ in range(len(A))]
    for i, j in enumerate(sort_map):
        reversed_A[j] = A[i]
    return reversed_A

# Generate constraint from each strong cover
def GenerateConstraintFromStrongCover(S, N, A, b):
    pi_0 = len(S) - 1
    pi = [0 for _ in range(len(N))]

    extended_S = ExtendSet(S)
    if pi_0 > 1:
        Sh_sum_old = SumFirstNCoeffsOfSet(S, A, 2)
        # TODO(jwd): Add in changing q
        for h in range(2, pi_0 + 1):
            Sh_sum_new = SumFirstNCoeffsOfSet(S, A, h + 1)
            Nh = set(i for i in N if A[i - 1] >= Sh_sum_old and
                     A[i -1] < Sh_sum_new)

            # Check validity
            if not CheckValidityOfCut(S, A, h, Nh, b):
                return None

            pi = SetCoefficientsForIndexSet(Nh, pi, h)
            extended_S -= Nh
            Sh_sum_old = Sh_sum_new

    # Check validity
    if not CheckValidityOfCut(S, A, 1, extended_S, b):
        return None
    pi = SetCoefficientsForIndexSet(extended_S, pi, 1)
    return pi, pi_0


# Generate output file
def WriteOutputFile(results_file, A, b, constraints, sort_map):
    f = open(results_file, 'w')
    f.write('ORIGINAL CONSTRAINT\n')
    f.write('\n')
    f.write('%d\n' % len(A))
    A = ReverseSortMap(A, sort_map)
    f.write('%s\n' % ','.join(str(a) for a in A))
    f.write('%d\n' % b)
    f.write('\n')
    f.write('--------------------\n')
    f.write('\n')
    f.write('NEW CONSTRAINTS\n')
    f.write('\n')
    f.write('--------------------\n')
    f.write('\n')

    for coefficients, rhs in constraints:
        coefficients = ReverseSortMap(coefficients, sort_map)
        f.write('%s\n' % ','.join(str(a) for a in coefficients))
        f.write('%d\n' % rhs)
        f.write('\n')

    f.close()

parser = argparse.ArgumentParser(description=('Reduce knapsack constraint to'
                                              'convex hull of integer points'))
parser.add_argument('input_file', help='the problem data file to process')
parser.add_argument('-r', '--results_file', default='results.dat',
                    help='name of results file (default: results.dat)')
args = parser.parse_args()
input_file = args.input_file
results_file = args.results_file

# Main routine (to be moved)
A, b = ReadInputFile(input_file)
N, A, sort_map = ConstructOrderedSet(A, b)
sets = GenerateMinimalCovers(N, A, b)
constraints = []
for S in sets:
    result = GenerateConstraintFromStrongCover(S, N, A, b)
    if result:
        constraints.append((result[0], result[1]))
WriteOutputFile(results_file, A, b, constraints, sort_map)
print 'Total time taken', time.clock() - t_
