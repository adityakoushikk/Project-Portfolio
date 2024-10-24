#Modules for randomization, graphing, and mean
import random
import matplotlib.pyplot as plt
from numpy import mean
from scipy.stats import binom 
 

#Post-mutated Sequence (Target Sequence; After Mutation)
a = str('VLYQGVNCT')
#X-Value on Graph
rX = []
#Y-Value on Graph
cY = []
#Loop runs 1000 times to create a monte carlo simulation
for count in range (0,1000):
    rX.append(count)
    #Second loop to calculate the number of cycles for pre-mutated sequence to become new sequence (post-mutation) 
    for count in range (0,100):
        #Goes through each letter in the target sequence
        for word in a.split():
            #random number to decide which amino acid is altered
            randomPlacement1 = int(random.randint(0,8))
            #Pre-mutated sequence (before mutation)
            words = ['V','L','Y','Q','D','V','N','C','T']
            #Change the amino acid chosen by the random Number into a new amino acid
            words[randomPlacement1] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            #Compile the new list into a single sequence
            new_sequence = str(words[0] + words[1] + words[2] + words[3] + word[4] + words[5] + words[6] + words[7] + words[8])
        #If the new sequence matches the Target sequence, print the number of cycles required to do so
        if new_sequence == a:
            #Append the cycle number to the Y value on the graph
            cY.append(count)
            #stop the loop once a match is found
            break
        
#Take average of all cycles
avg = mean(cY)
print(round(avg,2))

# setting the values 
# of n and p 
n = 2
p = 0.0055
# defining list of r values 
r_values = list(range(n + 1)) 
# list of pmf values 
dist = [binom.pmf(r, n, p) for r in r_values ] 
# plotting the graph 
plt.bar(r_values, dist) 
plt.show()

#Plot X and Y values
plt.plot(rX,cY)
plt.xlabel('Number of Runs')
plt.ylabel('Cycles required to change D614 to G per run')
plt.show()



















