
%HEADER
'File created by mtool program'

%HEADER.ANALYSIS
'plane_stress'

%NODE
8

%NODE.COORD
8
1	0.000000	1.000000	0.000000
2	0.500000	1.000000	0.000000
3	1.000000	1.000000	0.000000
4	0.000000	0.500000	0.000000
5	1.000000	0.500000	0.000000
6	1.000000	0.000000	0.000000
7	0.000000	0.000000	0.000000
8	0.500000	0.000000	0.000000

%NODE.SUPPORT
2
7	1	1	0	0	0	0
6	0	1	0	0	0	0

%MATERIAL
1

%MATERIAL.LABEL
1
1	'mymat'

%MATERIAL.ISOTROPIC
1
1	100.000000	0.250000

%INTEGRATION.ORDER
5
1	2	2	1	2	2	1
2	2	2	1	2	2	1
3	2	2	1	2	2	1
4	1	1	1	1	1	1
5	2	2	1	2	2	1

%ELEMENT
1

%ELEMENT.Q8
1
1	1	0	5	3	2	1	4	7	8	6	5

%LOAD
1
1	'Load_Case_1'

%LOAD.CASE
1

%LOAD.CASE.NODAL.FORCES
1
3	10.000000	5.000000	0.000000	0.000000	0.000000	0.000000

%END
