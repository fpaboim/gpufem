
%HEADER
'Sigma3d v2.0'

%HEADER.ANALYSIS
'plane_stress'

%NODE
80

%NODE.COORD
80
1	    9.000000     0.000000     0.000000
2	    9.000000     1.000000     0.000000
3	    9.000000     0.000000     1.000000
4	    9.000000     1.000000     1.000000
5	    9.000000     0.000000     2.000000
6	    9.000000     1.000000     2.000000
7	    9.000000     0.000000     3.000000
8	    9.000000     1.000000     3.000000
9	    8.000000     1.000000     0.000000
10	    8.000000     0.000000     0.000000
11	    8.000000     1.000000     1.000000
12	    8.000000     0.000000     1.000000
13	    8.000000     1.000000     3.000000
14	    8.000000     0.000000     3.000000
15	    8.000000     1.000000     2.000000
16	    8.000000     0.000000     2.000000
17	    7.000000     0.000000     3.000000
18	    7.000000     1.000000     3.000000
19	    7.000000     0.000000     2.000000
20	    7.000000     1.000000     2.000000
21	    7.000000     1.000000     0.000000
22	    7.000000     0.000000     0.000000
23	    7.000000     1.000000     1.000000
24	    7.000000     0.000000     1.000000
25	    5.000000     1.000000     3.000000
26	    6.000000     1.000000     3.000000
27	    6.000000     0.000000     3.000000
28	    5.000000     0.000000     3.000000
29	    6.000000     0.000000     2.000000
30	    6.000000     1.000000     2.000000
31	    5.000000     0.000000     2.000000
32	    5.000000     1.000000     2.000000
33	    6.000000     0.000000     0.000000
34	    6.000000     0.000000     1.000000
35	    6.000000     1.000000     0.000000
36	    6.000000     1.000000     1.000000
37	    5.000000     1.000000     1.000000
38	    5.000000     1.000000     0.000000
39	    5.000000     0.000000     1.000000
40	    5.000000     0.000000     0.000000
41	    4.000000     0.000000     0.000000
42	    4.000000     1.000000     0.000000
43	    4.000000     1.000000     1.000000
44	    4.000000     0.000000     1.000000
45	    4.000000     0.000000     2.000000
46	    4.000000     1.000000     2.000000
47	    4.000000     0.000000     3.000000
48	    4.000000     1.000000     3.000000
49	    3.000000     0.000000     3.000000
50	    3.000000     1.000000     3.000000
51	    3.000000     0.000000     2.000000
52	    3.000000     1.000000     2.000000
53	    3.000000     1.000000     0.000000
54	    3.000000     0.000000     0.000000
55	    3.000000     1.000000     1.000000
56	    3.000000     0.000000     1.000000
57	    2.000000     1.000000     3.000000
58	    2.000000     0.000000     3.000000
59	    2.000000     0.000000     2.000000
60	    2.000000     1.000000     2.000000
61	    2.000000     1.000000     0.000000
62	    2.000000     0.000000     0.000000
63	    2.000000     1.000000     1.000000
64	    2.000000     0.000000     1.000000
65	    1.000000     1.000000     3.000000
66	    1.000000     0.000000     3.000000
67	    0.000000     1.000000     3.000000
68	    0.000000     0.000000     3.000000
69	    0.000000     1.000000     2.000000
70	    0.000000     0.000000     2.000000
71	    1.000000     0.000000     2.000000
72	    1.000000     1.000000     2.000000
73	    1.000000     1.000000     1.000000
74	    1.000000     1.000000     0.000000
75	    1.000000     0.000000     1.000000
76	    0.000000     0.000000     0.000000
77	    0.000000     1.000000     0.000000
78	    0.000000     0.000000     1.000000
79	    1.000000     0.000000     0.000000
80	    0.000000     1.000000     1.000000

%ELEMENT
25

%ELEMENT.BRICK8
25
1     0     0    10     9     2     1    12    11     4     3 
2     0     0    22    21     9    10    24    23    11    12 
3     0     0    33    35    21    22    34    36    23    24 
4     0     0    40    38    35    33    39    37    36    34 
5     0     0    12    11     4     3    16    15     6     5 
6     0     0    24    23    11    12    19    20    15    16 
7     0     0    34    36    23    24    29    30    20    19 
8     0     0    39    37    36    34    31    32    30    29 
9     0     0    16    15     6     5    14    13     8     7 
10     0     0    19    20    15    16    17    18    13    14 
11     0     0    29    30    20    19    27    26    18    17 
12     0     0    31    32    30    29    28    25    26    27 
13     0     0    54    53    42    41    56    55    43    44 
14     0     0    62    61    53    54    64    63    55    56 
15     0     0    79    74    61    62    75    73    63    64 
16     0     0    76    77    74    79    78    80    73    75 
17     0     0    56    55    43    44    51    52    46    45 
18     0     0    64    63    55    56    59    60    52    51 
19     0     0    75    73    63    64    71    72    60    59 
20     0     0    78    80    73    75    70    69    72    71 
21     0     0    51    52    46    45    49    50    48    47 
22     0     0    59    60    52    51    58    57    50    49 
23     0     0    71    72    60    59    66    65    57    58 
24     0     0    70    69    72    71    68    67    65    66 
25     0     0    41    42    38    40    44    43    37    39 

%NODE.SUPPORT
8
76	1	1	1	0	0	0
67	1	0	0	0	0	0
68	1	0	0	0	0	0
69	1	0	0	0	0	0
70	1	0	0	0	0	0
77	1	0	0	0	0	0
78	1	0	0	0	0	0
80	1	0	0	0	0	0

%END