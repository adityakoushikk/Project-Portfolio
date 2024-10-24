import random
import matplotlib
import matplotlib.pyplot as plt
from numpy import mean

a = str('IYQASSTPC')
rX = []
cY = []

for count in range (1,1001):
    rX.append(count)
    for count in range (1,101):
        for word in a.split():
            randomPlacement1 = int(random.randint(0,8))
            words = ['I','Y','Q','A','G','S','T','P','C']
            words[randomPlacement1] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            bruh = str(words[0] + words[1] + words[2] + words[3] + word[4] + words[5] + words[6] + words[7] + words[8])
        if bruh == a:
            cY.append(count)
            break
avg = mean(cY)
print(round(avg,2))





plt.plot(rX,cY)
plt.xlabel('Number of Runs')
plt.ylabel('Cycles required to change D614 to G per run')
plt.show()
