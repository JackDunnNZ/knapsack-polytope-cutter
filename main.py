import time
import itertools
import argparse
import numpy
import pulp


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


def ReverseSortMap(A, sort_map):
    """Restores the order of a list to the order in the input file."""

    reversed_A = [0 for _ in range(len(A))]
    for i, j in enumerate(sort_map):
        reversed_A[j] = A[i]
    return reversed_A


def FindMJ(J, A, b):
    """Depth-first search with backtrack to find all subsets M of J."""

    n = len(J)
    A_J = [A[j - 1] for j in sorted(J)]
    sort_J = sorted(J)
    s = [0 for _ in range(n)]
    k = 0
    sets = []
    while k < n:
        # Current value of selected set
        v = SumCoeffsOverIndexList(s, A_J)
        s[k] += 1
        #print s, v + A_J[k]
        # Check if adding next variable creates cover
        if v + A_J[k] <= b:

            # Record current set as feasible M and reset current variable
            subset = set(sort_J[i] for i in range(n) if s[i])
            sets.append(subset)
        else:
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


def GenerateWConstraint(M, S, NS, A, b):
    """For a given M and S, solves a knapsack problem to find a constraint
    defining the polyhedron W."""

    prob = pulp.LpProblem("constraint_prob", pulp.LpMaximize)
    sortedS = sorted(S)
    y_vars = pulp.LpVariable.dicts('y', sortedS, 0, 1, 'Integer')
    rhs = b - SumCoeffsOverSet(M, A)
    prob += pulp.lpSum([y_vars[i] for i in sortedS]), "Objective"
    prob += pulp.lpSum(y_vars[i] * A[i - 1] for i in sortedS) <= rhs, "Con"
    prob.solve()
    z_M = pulp.value(prob.objective)

    Y_M = len(S) - 1 - z_M
    con = [1 if j in M else 0 for j in NS]
    return con, Y_M


def SimultaneousLiftingPulp(N, S, A, b):
    """For a given minimal cover generates all lifted constraint facets."""

    NS = N - S
    sorted_NS = sorted(NS)
    MJ = FindMJ(NS, A, b)

    # Generate constraints defining the polyhedron W
    con_list = []
    rhs_list = []
    for M in MJ:
        con, rhs = GenerateWConstraint(M, S, NS, A, b)
        con_list.append(con)
        rhs_list.append(rhs)

    # Prepare constraint set of W for reduction
    A_M = numpy.array(con_list)
    r = numpy.linalg.matrix_rank(A_M)
    if r == 0:
        return []
    b_M = numpy.array(rhs_list)
    zero_x = set()

    # Filter out variables forced to zero
    # For any constraint with a RHS of zero, any variables with a non-zero
    # coefficient must take on the value zero. These columns can then be
    # removed from the constraint set
    changed = True
    while changed:
        changed = False
        cols = set(range(len(A_M[0])))
        for i, b_i in enumerate(b_M):
            if b_i == 0:
                for j, A_Mi in enumerate(A_M[i, :]):
                    if A_Mi != 0:
                        changed = True
                        cols.discard(j)
                        zero_x.add(sorted_NS[j])
        cols = tuple(sorted(cols))
        if cols:
            A_M = A_M[:, cols]
        else:
            return []

    # Filter duplicate rows
    # Any two rows with the same constraint coefficients can be replaced with
    # a single constraint instead. The RHS takes the smaller of the two
    # original RHS values.
    for i in reversed(range(len(b_M))):
        remove = False
        con1 = A_M[i, :]
        for j in reversed(range(len(b_M))):
            if j == i:
                continue
            con2 = A_M[j, :]
            if numpy.array_equal(con1, con2) and b_M[i] >= b_M[j]:
                remove = True
                break
        if remove:
            A_M = numpy.delete(A_M, i, 0)
            b_M = numpy.delete(b_M, i, 0)

    # Find extreme points of W by solving the constraint system for all
    # possible basis combinations
    x_solns = []
    if A_M.any():
        r = numpy.linalg.matrix_rank(A_M)
        if r < len(A_M[0]):
            # No extreme points
            return []
        for comb in itertools.combinations(range(len(b_M)), r):
            A_Mc = numpy.array(A_M[comb, :])
            if numpy.linalg.matrix_rank(A_Mc) == r:
                b_Mc = numpy.array([rhs_list[i] for i in comb])
                x_c = numpy.linalg.solve(A_Mc, b_Mc)
                b_c = A_M.dot(numpy.transpose(x_c))
                if not sum(numpy.greater(b_c, b_M)):
                    x_solns.append(list(x_c))

    # Remove duplicate extreme points
    x_solns = sorted(x_solns)
    x_solns = list(k for k, _ in itertools.groupby(x_solns))

    # Generate facets from extreme points
    constraints = []
    rhs = len(S) - 1.0
    for x in x_solns:
        a_x = [0 for _ in range(len(A))]
        for i, y in enumerate(x):
            a_x[sorted_NS[i] - 1] = y
        for y in S:
            a_x[y - 1] = 1.0
        for y in zero_x:
            a_x[y - 1] = 0.0
        constraints.append([a_x, rhs])

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
        f.write('%s\n' % str(rhs))
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


def FindAllConstraints(N, A, b, used_sets):
    """For a given constraint and RHS, find all facets.

    Used sets is a set containing frozensets for each minimal cover that has
    already been processed.
    """

    sets = GenerateMinimalCovers(N, A, b)

    # Loop through all unused minimal covers to find new constraints
    constraints = []
    for S in sets:
        if frozenset(S) not in used_sets:
            result = SimultaneousLiftingPulp(N, S, A, b)
            if result:
                constraints += result
            used_sets.add(frozenset(S))
    return constraints, used_sets


def ComplementConstraint(A, b, C):
    """Complements a constraint and RHS for a complementing set C."""

    A_C = A[:]
    b_C = b
    for c in C:
        A_C[c - 1] *= -1
        b_C -= A[c - 1]
    return A_C, b_C


def UncomplementConstraint(A_C, b_C, C):
    """Uncomplements a constraint found for a complementing set C."""

    A = A_C[:]
    b = b_C
    for c in C:
        A[c - 1] *= -1
        b -= A_C[c - 1]
    return A, b


def main(imput_file, results_file, ampl_file):
    """Main solution routine. Finds all facets for inpu constraint."""

    t_ = time.clock()

    A, b, c = ReadInputFile(input_file)
    N, A, sort_map = ConstructOrderedSet(A)

    constraints = []
    used_sets = set()

    # Loop through all possible complementing sets
    for r in range(len(N) + 1):
        for C in itertools.combinations(N, r):

            # Complement set, find constraints and uncomplement
            A_C, b_C = ComplementConstraint(A, b, C)
            comp_constraints, used_sets = FindAllConstraints(N, A_C, b_C,
                                                             used_sets)
            new_constraints = []
            for a_C, a0_C in comp_constraints:
                a, a0 = UncomplementConstraint(a_C, a0_C, C)
                new_constraints.append([a, a0])
            constraints += new_constraints

    # Sort and filter constraints
    con_dict = {}
    for a in constraints:
        if not con_dict.get(str(a)):
            con_dict[str(a)] = a
    constraints = sorted(con_dict.values(), key=lambda a: (a[1], a[0]))

    # Write outputs
    WriteOutputFile(results_file, A, b, constraints, sort_map)
    WriteAmplDataFile(ampl_file, A, b, c, constraints, sort_map)
    print 'Total time taken', time.clock() - t_


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

    main(input_file, results_file, ampl_file)
