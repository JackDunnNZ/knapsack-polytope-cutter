import time
import itertools
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

    The fourth line contains the value of each variable, where the total value
    is to be maximised.

    Inputs:
        filename: A string, the filename of the input file to read.

    Returns:
        A list, the coefficients for the knapsack constraint.
        A number, the value for the right hand side in the knapsack constraint.
        A list, the value of each variable.
    """

    f = open(filename, 'r')

    # Read number of coefficients
    n = int(f.readline().rstrip('\n'))

    # Read in n coefficients
    A = [int(a) for a in f.readline().rstrip('\n').split(',')[0:n]]

    # Read RHS
    b = int(f.readline().rstrip('\n'))

    # Read in n entries for the value list
    c = [int(a) for a in f.readline().rstrip('\n').split(',')[0:n]]

    f.close()
    return A, b, c


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
    search with backtracking.

    Inputs:
        N: A set, indexing the variables of the constraint.
        A: A list, the corresponding coefficients of the constraint.
        b: A number, the right hand side of the constraint.

    Returns:
        A list of sets, each corresponding to a minimal cover.
    """

    n = len(N)
    sets = []

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
            sets.append(subset)
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

    return sets


def GenerateStrongCovers(minimal_covers):
    """Filter out all non-strong covers."""
    strong_sets = []

    # Sets classified by cardinality of set
    set_map = defaultdict(list)
    for subset in minimal_covers:
        set_map[len(subset)].append(subset)

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
        strong_sets += subsets

    return strong_sets


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


def GenerateQthConstraintFromStrongCover(S, N, A, b, q):
    """Generates a cutting constraint from a strong cover S, given q."""

    pi_0 = len(S) - 1
    pi = [0 for _ in range(len(N))]

    extended_S = ExtendSet(S)
    if q > 1:
        Sh_sum_old = SumFirstNCoeffsOfSet(S, A, 2)
        for h in range(2, q + 1):
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


def GenerateOneConstraintFromStrongCover(S, N, A, b):
    """Generates maximal constraint for a strong cover S."""

    constraints = []
    end = max(len(S) - 1, 1)
    result = GenerateQthConstraintFromStrongCover(S, N, A, b, end)
    if result:
        constraints.append((result[0], result[1]))
    return constraints


def GenerateBetaDashFromMinCover(S, A, b):
    """Generate all beta_dash coefficients for a minimal cover S."""

    beta_dash = ['' for _ in range(len(A))]
    h = 0
    i = len(A) - 1
    sortedS = sorted(S)

    sumSh = SumCoeffsOverSet(S, A)
    sumSh_1 = sumSh - A[sortedS[h] - 1]

    while i >= 0:
        if i + 1 in S:
            beta_dash[i] = 1
            i -= 1
        else:
            val = b - A[i]
            if val >= sumSh_1 and val < sumSh:
                beta_dash[i] = h
                i -= 1
            else:
                h += 1
                sumSh = sumSh_1
                sumSh_1 = sumSh - A[sortedS[h] - 1]
    return beta_dash


def GeneratePiFromMinCover(S, A):
    """Generate all pi coefficients for a minimal cover S."""

    pi = ['' for _ in range(len(A))]
    q = len(S) - 1
    i = 0
    extS = ExtendSet(S)
    sortedS = sorted(S)
    h = q

    sumSh_1 = SumFirstNCoeffsOfSet(S, A, h + 1)
    sumSh = sumSh_1 - A[sortedS[h] - 1]

    while i < len(A):
        if i + 1 not in extS:
            pi[i] = 0
            i += 1
        else:
            if i + 1 in S or h == 1:
                pi[i] = 1
                i += 1
            else:
                val = A[i]
                if val < sumSh_1 and val >= sumSh:
                    pi[i] = h
                    i += 1
                else:
                    h -= 1
                    if h >= 1:
                        sumSh_1 = sumSh
                        sumSh = sumSh_1 - A[sortedS[h] - 1]
    return pi


def PartitionMinCoverToIAndJ(S, beta_dash, pi):
    """Determine I and J, where i is in I if beta_dash[i] == pi[i]."""
    I = set()
    J = set()
    for i in range(len(beta_dash)):
        if i + 1 not in S:
            if beta_dash[i] == pi[i]:
                I.add(i + 1)
            else:
                J.add(i + 1)
    return I, J


def FindFacetsFromJ(J, S, A, b, pi):
    """Finds a sequentially lifted facet from J if it exists."""

    if not J:
        return pi

    beta = [pi[i] if i + 1 not in J else '' for i in range(len(A))]
    beta_j = [0 for _ in J]
    q = 0
    Q = set([J[q]])
    i = J[q] - 1
    beta[i] = pi[i] + 1
    beta_j[q] = 1

    while q < len(J) - 1:
        q += 1
        i = J[q] - 1

        # Sort SuQ by B/A
        SuQ = S.union(Q)
        B_by_A = [(float(beta[k - 1]) / A[k - 1], k) for k in SuQ]
        sorted_B_by_A = sorted(B_by_A, reverse=True)

        # Find p
        val = b - A[i]
        p = 1
        sumP = A[sorted_B_by_A[p - 1][1] - 1]
        sumP_1 = sumP + A[sorted_B_by_A[p][1] - 1]
        while True:
            if val >= sumP and val < sumP_1:
                break
            else:
                p += 1
                if p == len(SuQ):
                    return None
                sumP = sumP_1
                sumP_1 = sumP + A[sorted_B_by_A[p][1] - 1]

        # Find z_bar
        sum_bj = sum(beta[sorted_B_by_A[j][1] - 1] for j in range(p))
        sum_aj = sum(A[sorted_B_by_A[j][1] - 1] for j in range(p))
        z_bar = sum_bj + sorted_B_by_A[p][0] * (val - sum_aj)

        # Use sufficient conditions to determine beta if possible
        if z_bar < len(S) - pi[i] - 1:
            beta[i] = pi[i] + 1
            beta_j[q] = 1
        else:
            if sum_bj == len(S) - pi[i] - 1:
                beta[i] = pi[i]
                beta_j[q] = 0
            else:
                print 'dunno lol'
    return beta


def GenerateConstraintsFromMinimalCover(S, N, A, b):
    """Generate all sequentially lifted facets from a minimal cover."""

    # if len(set([6,7,8,9]).intersection(S)) == 4:
    #     print 'hi'
    # else:
    #     return None

    # Compute beta_dash and pi for each variable
    beta_dash = GenerateBetaDashFromMinCover(S, A, b)
    pi = GeneratePiFromMinCover(S, A)

    # Split N\S into sets I and J
    I, J = PartitionMinCoverToIAndJ(S, beta_dash, pi)

    # Generate a sequentially lifted facet for each permutation of J
    constraints = {}
    for j in itertools.permutations(J):
        beta = FindFacetsFromJ(j, S, A, b, pi)
        if beta and not constraints.get(str(beta)):
            constraints[str(beta)] = [beta, len(S) - 1]

    # Filter duplicate facets before returning
    constraints = constraints.values()
    return constraints


def WriteOutputFile(results_file, A, b, constraints, sort_map):
    """Write results file, containing all new cuts, ordered as in the input
    file.
    """

    f = open(results_file, 'w')
    f.write('ORIGINAL CONSTRAINT\n')
    f.write('\n')
    f.write('%d\n' % len(A))
    A_ordered = ReverseSortMap(A, sort_map)
    f.write('%s\n' % ','.join(str(a) for a in A_ordered))
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


def WriteAmplDataFile(ampl_file, A, b, c, constraints, sort_map):
    """Write data file for use in AMPL model with new constraints."""

    f = open(ampl_file, 'w')
    f.write('set VARIABLES := %s;\n' % ' '.join(str(i + 1) for i in
            range(len(A))))
    f.write('set CONSTRAINTS := %s;\n' % ' '.join(str(i + 1) for i in
            range(len(constraints) + 1)))
    f.write('\n')
    f.write('param c :=\n')
    for i, j in enumerate(c):
        f.write('%2d  %2d\n' % (i + 1, j))
    f.write(' ;\n')
    f.write('\n')
    f.write('param b :=\n')
    f.write('%2d  %2d\n' % (1, b))
    for i, j in enumerate(constraints):
        f.write('%2d  %2d\n' % (i + 2, j[1]))
    f.write(' ;\n')
    f.write('\n')
    f.write('param a (tr) :\n')
    f.write('     %s :=\n' % '   '.join(str(i + 1) for i in
            range(len(A))))
    A_ordered = ReverseSortMap(A, sort_map)
    f.write('%2d   %s\n' % (1, '   '.join(str(a) for a in A_ordered)))
    for i, constraint in enumerate(constraints):
        coefficients = ReverseSortMap(constraint[0], sort_map)
        f.write('%2d   %s\n' % (i + 2, '   '.join(str(a) for a in
                coefficients)))
    f.write(' ;\n')


"""Main routine."""
if __name__ == '__main__':
    # Include and parse command line arguments
    parser = argparse.ArgumentParser(
        description=('Reduce knapsack constraint to convex hull of integer '
                     'points'))
    parser.add_argument('-i', '--input_file', default='example_problem.dat',
                        help='the problem data file to process')
    parser.add_argument('-r', '--results_file', default='results.txt',
                        help='name of results file (default: results.txt)')
    parser.add_argument('-a', '--ampl_file', default='knapsack.dat', help=
                        'name of AMPL file to write (default: knapsack.dat)')
    args = parser.parse_args()
    input_file = args.input_file
    results_file = args.results_file
    ampl_file = args.ampl_file

    # Main solution routine
    t_ = time.clock()

    A, b, c = ReadInputFile(input_file)
    N, A, sort_map = ConstructOrderedSet(A)
    sets = GenerateMinimalCovers(N, A, b)
    print 'Covers generated', time.clock() - t_

    # Filtering out non-strong covers seems to impact run-time
    # sets = GenerateStrongCovers(sets)
    constraints = []
    for S in sets:
        result = GenerateConstraintsFromMinimalCover(S, N, A, b)
        if result:
            constraints += result

    # Filter any duplicate constraints
    con_dict = dict([(str(i), i) for i in constraints])
    constraints = sorted(con_dict.values(), key=lambda a: (a[1], a[0]))

    WriteOutputFile(results_file, A, b, constraints, sort_map)
    WriteAmplDataFile(ampl_file, A, b, c, constraints, sort_map)
    print 'Total time taken', time.clock() - t_
