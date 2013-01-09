
%HEADER
'File created by mtool program'

%HEADER.ANALYSIS
'plane_stress'

%NODE
16

%NODE.COORD
16
1	0.000000	2.000000	0.000000
2	0.000000	1.333333	0.000000
3	0.000000	0.666667	0.000000
4	0.000000	0.000000	0.000000
5	0.666667	1.333333	0.000000
6	0.666667	2.000000	0.000000
7	0.666667	0.666667	0.000000
8	0.666667	0.000000	0.000000
9	1.333333	2.000000	0.000000
10	2.000000	2.000000	0.000000
11	1.333333	0.666667	0.000000
12	1.333333	1.333333	0.000000
13	1.333333	0.000000	0.000000
14	2.000000	1.333333	0.000000
15	2.000000	0.000000	0.000000
16	2.000000	0.666667	0.000000

%NODE.SUPPORT
2
4	1	1	0	0	0	0
15	0	1	0	0	0	0

%MATERIAL
1

%MATERIAL.LABEL
1
1	'mymat'

%MATERIAL.ISOTROPIC
1
1	100.000000	0.250000

%MATERIAL.PROPERTY.DENSITY
1
1	10.000000

%INTEGRATION.ORDER
5
1	1	1	1	1	1	1
2	2	2	1	2	2	1
3	2	2	1	2	2	1
4	2	2	1	2	2	1
5	2	2	1	2	2	1

%ELEMENT
9

%ELEMENT.Q4
9
1	1	0	5	4	8	7	3
2	1	0	5	8	13	11	7
3	1	0	5	13	15	16	11
4	1	0	5	11	16	14	12
5	1	0	5	3	7	5	2
6	1	0	5	7	11	12	5
7	1	0	5	2	5	6	1
8	1	0	5	12	14	10	9
9	1	0	5	5	12	9	6

%LOAD
1
1	'Load_Case_1'

%LOAD.CASE
1

%LOAD.CASE.NODAL.FORCES
1
10	10.000000	0.000000	0.000000	0.000000	0.000000	0.000000

%END