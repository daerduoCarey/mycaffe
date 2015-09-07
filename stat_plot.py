#!/usr/bin

import re
import sys
import matplotlib.pyplot as plt

pat = re.compile('(?<=Iteration )[0-9]*')

USAGE = 'USAGE: stat_plot.py [keyword] [test_interval] file_1 file_2 ... file_k'

if len(sys.argv) < 4:
    print USAGE
    quit()

pat = re.compile('(?<= '+sys.argv[1]+' = ).*')

plt.xlabel('Iteration Number')
plt.ylabel(sys.argv[1])

test_interval = int(sys.argv[2])

color = ['b', 'r', 'm', 'y', 'g']

for i in range(3, len(sys.argv)):
    
    filename = sys.argv[i]
    file = open(filename, 'r')

    ite = []
    stat = []

    k = 0
    if filename == 'inception_6_bn_final':
        k = 32000

    for line in file:
        k = k + test_interval
        ite.append(k)
        l = pat.findall(line)
        value = 0
        if len(l) == 0:
            print "WARNING: no value found in this line!"
        if len(l) > 0:
            value = float(l[0])
        if len(l) > 1:
            print 'WARNING: more than one value found in each line!'
        stat.append(value)

    plt.plot(ite, stat, color[i-2], label=sys.argv[i])
    file.close()

plt.legend(loc=1)
plt.savefig('res.png', dpi=100)
plt.close()
