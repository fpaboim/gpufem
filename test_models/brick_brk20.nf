
%NODE
20

%NODE.COORD
20
1	      1.000000	      0.000000	      0.000000
2	      0.000000	      1.000000	      0.000000
3	      0.000000	      0.000000	      0.500000
4	      0.000000	      1.000000	      0.500000
5	      1.000000	      1.000000	      0.500000
6	      1.000000	      0.000000	      0.500000
7	      0.500000	      0.000000	      1.000000
8	      0.000000	      0.500000	      1.000000
9	      0.500000	      1.000000	      1.000000
10	      1.000000	      0.500000	      1.000000
11	      0.500000	      0.000000	      0.000000
12	      0.000000	      0.500000	      0.000000
13	      0.500000	      1.000000	      0.000000
14	      1.000000	      0.500000	      0.000000
15	      0.000000	      0.000000	      1.000000
16	      0.000000	      1.000000	      1.000000
17	      1.000000	      1.000000	      1.000000
18	      1.000000	      0.000000	      1.000000
19	      0.000000	      0.000000	      0.000000
20	      1.000000	      1.000000	      0.000000

%NODE.SUPPORT
13
1	1	1	1	1	1	1
16	1	1	1	1	1	1
19	1	1	1	1	1	1
8	1	1	1	1	1	1
7	1	1	1	1	1	1
9	1	1	1	1	1	1
17	1	1	1	1	1	1
18	1	1	1	1	1	1
11	1	1	1	1	1	1
3	1	1	1	1	1	1
10	1	1	1	1	1	1
15	1	1	1	1	1	1
6	1	1	1	1	1	1

%LOAD.CASE.NODAL.FORCE
1
20	0	0	10	0	0	0

%INTEGRATION.ORDER
2
1	2	2	2	2	2	2
2	2	2	1	2	2	1

%ELEMENT
1

%ELEMENT.BRICK20
1
1	0	1	1	11	19	12	2	13	20	14	6	3	4	5	18	7	15	8	16	9	17	10

%END
