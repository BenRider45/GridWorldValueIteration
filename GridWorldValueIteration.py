

#An Experiment to try an implement gridworld value iteration wiht an example problem



import numpy as np

import copy
## Constants
GAMMA = 0.8

#Paramaters
##ROWS: number of rows in grid
##COLS: number of columns in grid
##initValue: either an int, or array of shape (rows,cols) which defines the intiial value of each gridspace
def BuildGrid(ROWS,COLS,initValue):

    #defining states, using a 2d array, with each index storing the states current value

    
    if(type(initValue)==int):
        grid = np.full((ROWS,COLS),initValue)
    elif(initValue.shape == (ROWS,COLS)):
        grid = np.empty((ROWS,COLS))
        for i in range(ROWS):
            for j in range(COLS):
                grid[i][j] = initValue[i][j]
        
    return grid

## Modify this to have ROWS*COLS entries, number is immediate reward from action
#T=Terminal State
actions = {
    (0,0):([("D",0),("R",0)]),
    (0,1):([("L",0),("D",0),("R",50)]),
    (0,2):([("T",0)]),
    (1,0):([("U",0),("R",0)]),
    (1,1):([("L",0),("U",0),("R",0)]),
    (1,2):([("L",0),("U",100)])
}

policy = {

    (0,0):("DR"),
    (0,1):("LDR"),
    (0,2):("T"),
    (1,0):("UR"),
    (1,1):("LUR"),
    (1,2):("LU")

}
##Uses Value Iteration Algorithm which assigns the value of our state to the highest value that can be attained from that state, 
# I.e choosing the action which maximizes value. rather than implementing Bellman-equality
def GreedyValueIteration(grid,actions,maxIter):
    newGrid = grid
    for k in range(maxIter):
        for j in range(grid.shape[1]):
            for i in range(grid.shape[0]):
                v = grid[i][j]
                possibleValues = []
                if actions[(i,j)][0][0]=="T":
                    grid[i][j] =0
                else:
                    for h in range(len(actions[i,j])):
                        if actions[i,j][h][0] == "D":
                            possibleValues.append(actions[(i,j)][h][1] + GAMMA*grid[i+1][j])
                        elif actions[i,j][h][0] == "U":
                            possibleValues.append(actions[(i,j)][h][1] + GAMMA*grid[i-1][j])
                        elif actions[i,j][h][0] == "R":
                            possibleValues.append(actions[(i,j)][h][1] + GAMMA*grid[i][j+1])
                        elif actions[i,j][h][0] == "L":
                            possibleValues.append(actions[(i,j)][h][1] + GAMMA*grid[i][j-1])
                    

                    newGrid[i][j] = max(possibleValues)
        

    return newGrid

## Computes the optimal policy based on given values for states
def computePolicy(grid,actions):
    newPolicy = copy.deepcopy(policy)
    for j in range(grid.shape[1]):
            for i in range(grid.shape[0]):
                possibleValues = [0]*4
                if actions[(i,j)][0][0]=="T":
                    policy[(i,j)] = "T"
                else:
                    for h in range(len(actions[i,j])):
                        if actions[i,j][h][0] == "D":
                            possibleValues[0]= actions[(i,j)][h][1] + GAMMA*grid[i+1][j]
                        elif actions[i,j][h][0] == "U":
                            possibleValues[1] = (actions[(i,j)][h][1] + GAMMA*grid[i-1][j])
                        elif actions[i,j][h][0] == "R":
                            possibleValues[2] = (actions[(i,j)][h][1] + GAMMA*grid[i][j+1])
                        elif actions[i,j][h][0] == "L":
                            possibleValues[3] = (actions[(i,j)][h][1] + GAMMA*grid[i][j-1])
                    
                    maxIndexes = [g for g, x in enumerate(possibleValues) if x == max(possibleValues)]
                    newPolicy[(i,j)] = ""
                    if 0 in maxIndexes:
                        newPolicy[(i,j)] +="D"
                    if 1 in maxIndexes:
                        newPolicy[(i,j)] +="U"
                    if 2 in maxIndexes:
                        newPolicy[(i,j)] += "R"
                    if 4 in maxIndexes:
                        newPolicy[(i,j)] += "L"
    return newPolicy

def main():
    ROWS = 2
    COLS = 3
    Grid = BuildGrid(ROWS,COLS,0)
    print(Grid)
    print(policy[0,1])
    FoundPolicy = policy
    print(FoundPolicy)
    for i in range(1,10):

        newGrid = GreedyValueIteration(Grid,actions,i)

        iterPolicy = computePolicy(newGrid,actions)
        
        print("========i: ",i,"=======")
        print(newGrid)
        print("LastPolicy = ",FoundPolicy)
        print("NewPolicy = ",iterPolicy)
        EQUAL = 1

        if(FoundPolicy==iterPolicy):
            print("Convergence after ",i," iterations with policy:")
            print(iterPolicy)
            break
        else:
            FoundPolicy = iterPolicy

    
main()