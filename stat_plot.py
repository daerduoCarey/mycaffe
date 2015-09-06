#!/usr/bin

import re
import sys
import matplotlib.pyplot as plt

pat = re.compile('(?<=Iteration )[0-9]*')

USAGE = 'USAGE: stat_plot.py [loss | acc] file_1 file_2 ... file_k'

if len(sys.argv) < 3:
    print USAGE
    quit()

iter_pat = re.compile('(?<=Iteration )[0-9]*')
loss_pat = re.compile('(?<=, loss = ).*')
acc_pat = re.compile('(?<= accuracy = ).*')

plt.xlabel('Iteration Number')
plt.ylabel('Loss')

color = ['b', 'r', 'm', 'y', 'g']

for i in range(2, len(sys.argv)):
    
    filename = sys.argv[i]
    file = open(filename, 'r')

    ite = []
    stat = []

    k = 0

    if sys.argv[1] == 'loss':
    
        for line in file:
            ite.append(int(iter_pat.findall(line)[0]))
            stat.append(float(loss_pat.findall(line)[0]))
    
    elif sys.argv[1] == 'acc':
        for line in file:
            k = k + 1
            ite.append(k)
            stat.append(float(acc_pat.findall(line)[0]))
    
    else:
        print USAGE
        quit()

    plt.plot(ite, stat, color[i-1], label=sys.argv[i])
    file.close()

plt.legend(loc=1)
plt.savefig('res.png', dpi=100)
plt.close()
