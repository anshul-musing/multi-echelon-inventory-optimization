
"""This module calls the multi-echelon supply chain simulation
as a black box function to optimize inventory policy
"""

__author__ = 'Anshul Agarwal'


from simulation.simLostSales import simulate_network
#from simulation.simBackorder import simulate_network
import numpy as np
from skopt import gp_minimize, forest_minimize
import csv
import time


# Read the historical demand data
demandAllNodes = []
with open('data/demandData.csv', 'r') as f:
    next(f)  # skip headings
    reader = csv.reader(f)
    for a in reader:
        demandAllNodes.append([float(j) for j in a])

demandAllNodes = np.array(demandAllNodes)  # contains all nodes except the source node

# Read the historical data on lead time delay
leadTimeDelay = []
with open('data/leadTimeExtraDays.csv', 'r') as f:
    reader = csv.reader(f)
    for a in reader:
        leadTimeDelay.append(int(a[0]))

leadTimeDelay = np.array(leadTimeDelay)

# Define the supply chain network
numNodes = 6
nodeNetwork = np.zeros((numNodes, numNodes))
nodeNetwork[0, 1] = 1
nodeNetwork[1, 2] = 1
nodeNetwork[1, 3] = 1
nodeNetwork[3, 4] = 1
nodeNetwork[3, 5] = 1

# Initialize network nodes
defaultLeadTime = np.array([0, 3, 4, 4, 2, 2])
serviceTarget = np.array([0.0, 0.95, 0.95, 0.0, 0.95, 0.95])


# function to evaluate the objective function for optimization
# we minimize on-hand inventory and heavily penalize not meeting
# the beta service level (demand volume based)

def getObj(initial_guess):

	# Split the initial guess to get base stock and ROP
    excess_inventory_guess = initial_guess[:(numNodes - 1)]
    ROP_guess = initial_guess[(numNodes - 1):]
    base_stock_guess = np.add(excess_inventory_guess, ROP_guess)
    
    # Insert the supply node's base stock
    baseStock = np.insert(base_stock_guess, 0, 10000)
    
    # Insert a zero ROP for the first source node
    ROP = np.insert(ROP_guess, 0, 0)
    
    # Initialize inventory level
    initialInv = 0.9*baseStock
    
    replications = 20
    totServiceLevel = np.zeros(numNodes)
    totAvgOnHand = 0.0
    for i in range(replications):
        nodes = simulate_network(i,numNodes,nodeNetwork,initialInv,ROP,baseStock,\
                                 demandAllNodes,defaultLeadTime,leadTimeDelay)
		
        totServiceLevel = np.array([totServiceLevel[j] + \
                                    nodes[j].serviceLevel for j in range(len(nodes))]) #convert list to array
		
        totAvgOnHand += np.sum([nodes[j].avgOnHand for j in range(len(nodes))])
    
    servLevelPenalty = np.maximum(0, serviceTarget - totServiceLevel/replications) # element-wise max
    objFunValue = totAvgOnHand/replications + 1.0e6*np.sum(servLevelPenalty)
    return objFunValue


# Callback function to print optimization iterations
niter = 1
def callbackF(res):
    global niter
    print('{0:4d}    {1:6.6f}'.format(niter, res.fun))
    niter += 1

######## Main statements to call optimization ########
excess_inventory_initial_guess = [2000, 350, 700, 150, 400]
ROP_initial_guess = [1000, 250, 200, 150, 200]
guess_vec = excess_inventory_initial_guess + ROP_initial_guess # concatenate lists
guess = []
for j in range(len(guess_vec)):
    g = (0, guess_vec[j])
    guess.append(g)

NUM_CYCLES = 1000
TIME_LIMIT = 1440 # minutes
start_time = time.time()
print("\nMax time limit: " + str(TIME_LIMIT) + " minutes")
print("Max algorithm cycles: " + str(NUM_CYCLES) + " (20 iterations per cycle)")
print("The algorithm will run either for " + str(TIME_LIMIT) + " minutes or " + str(NUM_CYCLES) + " cycles")
ctr = 1
elapsed_time = (time.time() - start_time)/60.0
bestObj = 1e7
bestSoln = []
bestCycle = 0
while ctr <= NUM_CYCLES and elapsed_time <= TIME_LIMIT:
    print('\nCycle: ' + str(ctr))
    print('{0:4s}    {1:9s}'.format('Iter', 'Obj'))
    """
    opt = forest_minimize(func=getObj
                        , dimensions=guess
                        , n_calls=20
                        , n_random_starts=10
                        , random_state=707
                        , verbose=False
                        , callback=callbackF
                        , kappa=50)
    """
    opt = gp_minimize(func=getObj
                        , dimensions=guess
                        , n_calls=20
                        , n_random_starts=10
                        , random_state=ctr
                        , verbose=False
                        , callback=callbackF
                        , kappa=50)

    if opt.fun < bestObj:
        bestObj = opt.fun
        bestSoln = opt.x
        bestCycle = ctr
    ctr += 1
    elapsed_time = (time.time() - start_time)/60.0

print("\nFinal objective: " + "{0:10.3f}".format(bestObj))
print("\nFinal solution: " + str(bestSoln))
print("\nBest cycle: " + str(bestCycle))
print("\nTotal time: " + "{0:3.2f}".format(elapsed_time) + " minutes")

"""
Backorder case

Final objective:   1516.884
Final solution: [1325, 196, 346, 99, 149, 709, 187, 181, 91, 188]
Best cycle: 192
Total time: 399.42 minutes

Lost sales case

Final objective:   1277.106
Final solution: [1122, 57, 235, 18, 8, 732, 195, 189, 106, 182]
Best cycle: 228
Total time: 387.40 minutes
"""
