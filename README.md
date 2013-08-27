Knapsack Polytope Cutter
========================

Produces a set of extra constraints to reduce the feasible region of a knapsack problem to the convex hull of feasible integer points. This ensures the LP relaxation will give integer values. It also allows the calculation of the IP dual variable through imputation of the LP relaxation's duals.

## Dependencies

Required packages:
* [PuLP](https://code.google.com/p/pulp-or/)
* [NumPy](http://www.numpy.org/)

## Usage

The script can be called using the following command (assuming `python` is on the system PATH):

    python main.py -i input_file -r results_file -a ampl_file

* `input_file` is the path to the input data file (as described below) - defaults to the included file `example_problem.dat`.
* `results_file` is the path to the output file that will be written - defaults to `results.txt`.
* `ampl_file` is the path to the ampl data file that will be written - defaults to `knapsack.dat`.

## Input File

The input file must be of a specified form. This is demonstrated in the included `example_problem.dat`. Note that the script assumes a maximisation problem with a <= constraint.

* The first line must contain the number of variables in the problem.
* The second line contains the coefficients of each variable in the knapsack constraint.
* The third line contains the right hand side of the knapsack constraint.
* The final line contains the objective coefficients of each variable.

## Output Files

The script produces two output files.

The first output is a text file containing the original problem and the new constraints, in the format of the input data file. The original problem is printed first, followed by a set of coefficients and a right hand side foe each new constraint (following the structure of the constraint in the original problem). These are all <= constraints.

The second output combines the original problem data and the new constraints into an AMPL data file that is ready to be solved. The files `knapsack.mod` and `knapsack.run` are the corresponding model and run files for this data file. If the default name `knapsack.dat` is used, the new model can be solved simply by running:

    ampl knapsack.run

If the default name is changed, this will also need to be altered in `knapsack.run`.
