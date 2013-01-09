
%HEADER
'File created by mtool program'

%HEADER.ANALYSIS
'plane_stress'

%NODE
21

%NODE.COORD
21
1	0.000000	0.500000	0.000000
2	0.000000	0.000000	0.000000
3	0.500000	0.000000	0.000000
4	0.500000	1.000000	0.000000
5	0.000000	1.000000	0.000000
6	0.500000	2.000000	0.000000
7	0.000000	2.000000	0.000000
8	0.000000	1.500000	0.000000
9	1.000000	1.000000	0.000000
10	1.000000	0.000000	0.000000
11	1.000000	0.500000	0.000000
12	2.000000	1.000000	0.000000
13	1.000000	2.000000	0.000000
14	1.000000	1.500000	0.000000
15	2.000000	1.500000	0.000000
16	2.000000	0.500000	0.000000
17	2.000000	0.000000	0.000000
18	2.000000	2.000000	0.000000
19	1.500000	2.000000	0.000000
20	1.500000	0.000000	0.000000
21	1.500000	1.000000	0.000000

%NODE.SUPPORT
2
2	1	1	0	0	0	0
17	0	1	0	0	0	0

%MATERIAL
1

%MATERIAL.LABEL
1
1	'mymat'

%MATERIAL.ISOTROPIC
1
1	1.000000	0.250000

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
4

%ELEMENT.Q8
4
1	1	0	2	10	11	9	4	5	1	2	3
2	1	0	2	17	16	12	21	9	11	10	20
3	1	0	2	9	14	13	6	7	8	5	4
4	1	0	2	12	15	18	19	13	14	9	21

%LOAD
1
1	'Load_Case_1'

%LOAD.CASE
1

%LOAD.CASE.NODAL.FORCES
1
18	10.000000	0.000000	0.000000	0.000000	0.000000	0.000000

%END
