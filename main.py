import time
import argparse
from collections import defaultdict


def ReadInputFile(filename):
    """Reads input file containing problem data.

    The first line of the input file contains the number of variables. All of
    these are assumed to be binary (0-1) variables.

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


def ConstructOrderedSet(A):
    """Sorts coefficient list into descending order.

    Inputs:
        A: A list, containing the coefficients to sort.

    Returns:
        An index set for the ordered coefficients.
        A list containing the ordered coefficients.
        A list containing a mapping from the sorted coefficients back to the
        original order.
    """

    sort_map = sorted(range(len(A)), key=A.__getitem__, reverse=True)
    A.sort(reverse=True)
    N = set(range(1, len(A) + 1))
    return N, A, sort_map


def SumCoeffsOverSet(summing_set, A):
    """Returns the sum of coefficients corresponding to the summing set."""

    return sum(A[i - 1] for i in summing_set)


def SumCoeffsOverIndexList(summing_list, A):
    """Returns the sum of the coefficient and summing lists mulitplied
    element-wise.
    """

    return sum(A[i] * summing_list[i] for i in range(len(summing_list)))


def ConvertIndexListToSet(index_list):
    """Creates a set containing the indices of all '1' entries in the index
    list
    """

    return set(i + 1 for i, j in enumerate(index_list) if j == 1)


def ExtendSet(index_set):
    """Creates the extension of an index set by adding to the set all elements
    with indices smaller than the smallest index in the index set.
    """

    min_elem = min(index_set)
    return index_set.union(set(range(1, min_elem)))


def GenerateMinimalCovers(N, A, b):
    """Finds all minimal covers of a knapsack constraint using a depth-first
    search with backtracking. Minimal covers are then filtered to leave only
    strong covers.

    Inputs:
        N: A set, indexing the variables of the constraint.
        A: A list, the corresponding coefficients of the constraint.
        b: A number, the right hand side of the constraint.

    Returns:
        A list of sets, each corresponding to a strong minimal cover.
    """

    # Sets classified by cardinality of set
    set_map = defaultdict(list)

    n = len(N)

    # Depth-first search with backtracking
    s = [0 for _ in range(n)]
    k = 0
    while k < n:
        # Current value of selected set
        v = SumCoeffsOverIndexList(s, A)
        s[k] += 1

        # Check if adding next variable creates cover
        if v + A[k] > b:

            # Record current set as minimum cover and reset variable
            subset = ConvertIndexListToSet(s)
            set_map[len(subset)].append(subset)
            s[k] = 0

        # Move to next variable
        k += 1

        # Check for backtracking
        if k == n:

            # Terminate if all combinations tested
            if v == 0:
                break

            # Backtrack to last selected variable and remove
            k -= 1
            s[k] = 0
            while s[k] != 1:
                s[k] = 0
                k -= 1
            s[k] = 0
            k += 1

    # Filter out all non-strong covers
    sets = []

    # Compare subsets of same cardinality only
    for subsets in set_map.itervalues():

        # If extension of set is subset of any other set extension, remove
        # this set from potential strong covers
        extended_subsets = [ExtendSet(ss) for ss in subsets]
        l = len(subsets) - 1
        for i in range(len(subsets)):
            for j in range(len(subsets)):
                if extended_subsets[l - i] < extended_subsets[l - j]:
                    subsets.pop(l-i)
                    break

        # Record all remaining sets as strong covers
        sets += subsets

    return sets


def SetCoefficientsForIndexSet(index_set, pi, value):
    """Set all entries of pi indicated by the index set equal to the value
    specified.
    """

    for i in index_set:
        pi[i - 1] = value
    return pi


def FirstNElemsOfSet(index_set, n):
    """Select the smallest n elements of an index set."""

    return set(sorted(list(index_set))[:n])


def SumFirstNCoeffsOfSet(index_set, A, n):
    """Sum the values of the n largest coefficients from those indicated by
    the index set.
    """

    reduced_set = FirstNElemsOfSet(index_set, n)
    return SumCoeffsOverSet(reduced_set, A)


def CheckValidityOfCut(S, A, h, Nh, b):
    """Checks whether a given inequality will correspond to a facet-defining
    cut.
    """

    # Reduce the set S to the first h+1 elements
    reduced_S = S - FirstNElemsOfSet(S, h + 1)
    reduced_sum = SumCoeffsOverSet(reduced_S, A)

    # If any extra element added violates the constraint, reject the constraint
    for i in Nh:
        if reduced_sum + A[i - 1] > b:
            return False
    return True


def ReverseSortMap(A, sort_map):
    """Restores the order of a list to the order in the input file."""

    reversed_A = [0 for _ in range(len(A))]
    for i, j in enumerate(sort_map):
        reversed_A[j] = A[i]
    return reversed_A


def GenerateConstraintFromStrongCover(S, N, A, b):
    """Generates a cutting constraint from a strong cover S."""

    pi_0 = len(S) - 1
    pi = [0 for _ in range(len(N))]

    extended_S = ExtendSet(S)
    if pi_0 > 1:
        Sh_sum_old = SumFirstNCoeffsOfSet(S, A, 2)
        # TODO(jwd): Add in changing q
        for h in range(2, pi_0 + 1):
            Sh_sum_new = SumFirstNCoeffsOfSet(S, A, h + 1)
            Nh = set(i for i in N if A[i - 1] >= Sh_sum_old and
                     A[i - 1] < Sh_sum_new)

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


def WriteOutputFile(results_file, A, b, constraints, sort_map):
    """Write results file, containing all new cuts, ordered as in the input
    file.
    """

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

"""Main routine."""
if __name__ == '__main__':
    # Include and parse command line arguments
    parser = argparse.ArgumentParser(
        description=('Reduce knapsack constraint to convex hull of integer '
                     'points'))
    parser.add_argument('input_file', help='the problem data file to process')
    parser.add_argument('-r', '--results_file', default='results.dat',
                        help='name of results file (default: results.dat)')
    args = parser.parse_args()
    input_file = args.input_file
    results_file = args.results_file

    # Main solution routine
    t_ = time.clock()

    A, b = ReadInputFile(input_file)
    N, A, sort_map = ConstructOrderedSet(A)
    sets = GenerateMinimalCovers(N, A, b)
    constraints = []
    for S in sets:
        result = GenerateConstraintFromStrongCover(S, N, A, b)
        if result:
            constraints.append((result[0], result[1]))
    WriteOutputFile(results_file, A, b, constraints, sort_map)
    print 'Total time taken', time.clock() - t_
