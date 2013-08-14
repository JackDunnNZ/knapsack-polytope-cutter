model;

set VARIABLES;
set CONSTRAINTS;

param a {VARIABLES, CONSTRAINTS};

param b {CONSTRAINTS};

param c {VARIABLES};

var x {VARIABLES} >=0, <=1;

maximize Value: sum {i in VARIABLES} c[i] * x[i];

subject to MeetConstraints {j in CONSTRAINTS} :
  sum {i in VARIABLES} a[i, j] * x[i] <= b[j];
