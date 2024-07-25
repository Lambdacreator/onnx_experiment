# import libraries
from pcraster import *
from pcraster.framework import *
import numpy as np
import os

# create folder to store data
path = r'/Users/cr/Documents/UU/Master/thesis/life' 
if not os.path.exists(path):
    os.makedirs(path)

# set path
os.chdir(r'/Users/cr/Documents/UU/Master/thesis/life')

####################################################################################################################################################################################################################################

"""
Training data and validation data - dymanic
"""
# create a map with random numbers from 0 to 1 for further process
class GameOfLife(DynamicModel):
  def __init__(self):
    DynamicModel.__init__(self)
    setclone('clone.map')

  def initial(self):
    setclone(20, 20, 1.0, 0, 0) # define size of the board
    
    pcraster._pcraster.setrandomseed(66) # set seed
    aUniformMap = uniform(1) # create random number map
    self.alive = aUniformMap < 0.15 
    ''''
    convert numbers: 
    if smaller than 0.15 -> alive
    else dead
    '''

    # report the alive cells as train data
    self.report(self.alive, 'traind')

  def dynamic(self):
    aliveScalar = scalar(self.alive)
    numberOfAliveNeighbours = windowtotal(aliveScalar, 3) - aliveScalar

    threeAliveNeighbours = numberOfAliveNeighbours == 3
    birth = pcrand(threeAliveNeighbours, pcrnot(self.alive))

    survivalA = pcrand((numberOfAliveNeighbours == 2), self.alive)
    survivalB = pcrand((numberOfAliveNeighbours == 3), self.alive)
    survival = pcror(survivalA, survivalB)
    self.alive = pcror(birth, survival)
    self.report(self.alive, 'traind')

nrOfTimeSteps = 19999
myModel = GameOfLife()
dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
dynamicModel.run()

# generate all filenames
filenames = []
for i in range(1, 20000):
    filename = f"{'traind'}{i // 1000:02d}.{i % 1000:03d}"
    filenames.append(filename)

# initialize lists
X_train, y_train, X_val, y_val = [], [], [], []

Xtrain = readmap('traind.map')
X_train.append(pcr2numpy(Xtrain, 2))


# process files for training data
for i, filename in enumerate(filenames[:15999]):
    if os.path.exists(filename): # ensure the file exists
        raster = readmap(filename) # read the raster file
        raster_np = pcr2numpy(raster, 2) # convert to NumPy array
        if i % 2 == 0:
            y_train.append(raster_np)
        else:
            X_train.append(raster_np)

# process files for validation data
for i, filename in enumerate(filenames[15999:]):
    if os.path.exists(filename): # ensure the file exists
        raster = readmap(filename) # read the raster file
        raster_np = pcr2numpy(raster, 2) # convert to NumPy array
        if i % 2 == 0:
            y_val.append(raster_np)
        else:
            X_val.append(raster_np)

# convert lists to NumPy arrays and reshape if necessary
X_train = np.array(X_train).reshape(-1, 20, 20, 1)
y_train = np.array(y_train).reshape(-1, 20, 20, 1)
X_val = np.array(X_val).reshape(-1, 20, 20, 1)
y_val = np.array(y_val).reshape(-1, 20, 20, 1)

# save the arrays as .npy files in 'life' folder
np.save('X_train_d.npy', X_train)
np.save('y_train_d.npy', y_train)
np.save('X_val_d.npy', X_val)
np.save('y_val_d.npy', y_val)

####################################################################################################################################################################################################################################

"""
Training data and validation data - static
"""
# create a map with random numbers from 0 to 1 for further process
class GameOfLife(DynamicModel):
  def __init__(self):
    DynamicModel.__init__(self)
    setclone('clone.map')

  def initial(self):
    setclone(20, 20, 1.0, 0, 0) # define size of the board
    
    pcraster._pcraster.setrandomseed(88) # set seed
    aUniformMap = uniform(1) # create random number map
    self.alive = aUniformMap < 0.15 
    ''''
    convert numbers: 
    if smaller than 0.15 -> alive
    else dead
    '''

    # report the alive cells as train data
    self.report(self.alive, 'trains')

  def dynamic(self):
    aliveScalar = scalar(self.alive)
    numberOfAliveNeighbours = windowtotal(aliveScalar, 3) - aliveScalar

    threeAliveNeighbours = numberOfAliveNeighbours == 3
    birth = pcrand(threeAliveNeighbours, pcrnot(self.alive))

    survivalA = pcrand((numberOfAliveNeighbours == 2), self.alive)
    survivalB = pcrand((numberOfAliveNeighbours == 3), self.alive)
    survival = pcror(survivalA, survivalB)
    self.alive = pcror(birth, survival)
    self.report(self.alive, 'trains')

nrOfTimeSteps = 19999
myModel = GameOfLife()
dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
dynamicModel.run()

# generate all filenames
filenames = []
for i in range(1, 20000):
    filename = f"{'trains'}{i // 1000:02d}.{i % 1000:03d}"
    filenames.append(filename)

# initialize lists
X_train, y_train, X_val, y_val = [], [], [], []

Xtrain = readmap('trains.map')
X_train.append(pcr2numpy(Xtrain, 2))


# process files for training data
for i, filename in enumerate(filenames[:15999]):
    if os.path.exists(filename): # ensure the file exists
        raster = readmap(filename) # read the raster file
        raster_np = pcr2numpy(raster, 2) # convert to NumPy array
        if i % 2 == 0:
            y_train.append(raster_np)
        else:
            X_train.append(raster_np)

# process files for validation data
for i, filename in enumerate(filenames[15999:]):
    if os.path.exists(filename): # ensure the file exists
        raster = readmap(filename) # read the raster file
        raster_np = pcr2numpy(raster, 2) # convert to NumPy array
        if i % 2 == 0:
            y_val.append(raster_np)
        else:
            X_val.append(raster_np)

# convert lists to NumPy arrays and reshape if necessary
X_train = np.array(X_train).reshape(-1, 20, 20, 1)
y_train = np.array(y_train).reshape(-1, 20, 20, 1)
X_val = np.array(X_val).reshape(-1, 20, 20, 1)
y_val = np.array(y_val).reshape(-1, 20, 20, 1)

# save the arrays as .npy files in 'life' folder
np.save('X_train_s.npy', X_train)
np.save('y_train_s.npy', y_train)
np.save('X_val_s.npy', X_val)
np.save('y_val_s.npy', y_val)

####################################################################################################################################################################################################################################

"""
Testing data
"""
# create a new initial map and report it as test data
class GameOfLife(DynamicModel):
  def __init__(self):
    DynamicModel.__init__(self)
    setclone('clone.map')

  def initial(self):
    setclone(20, 20, 1.0, 0, 0) # define size of the board
    
    pcraster._pcraster.setrandomseed(888) # set seed
    aUniformMap = uniform(1) # create random number map
    self.alive = aUniformMap < 0.15 
    ''''
    convert numbers: 
    if smaller than 0.15 -> alive
    else dead
    '''

    # report the alive cells as train data
    self.report(self.alive, 'test')

  def dynamic(self):
    aliveScalar = scalar(self.alive)
    numberOfAliveNeighbours = windowtotal(aliveScalar, 3) - aliveScalar

    threeAliveNeighbours = numberOfAliveNeighbours == 3
    birth = pcrand(threeAliveNeighbours, pcrnot(self.alive))

    survivalA = pcrand((numberOfAliveNeighbours == 2), self.alive)
    survivalB = pcrand((numberOfAliveNeighbours == 3), self.alive)
    survival = pcror(survivalA, survivalB)
    self.alive = pcror(birth, survival)
    self.report(self.alive, 'test')

nrOfTimeSteps = 99
myModel = GameOfLife()
dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
dynamicModel.run()

# generate all filenames
filenames = []
for i in range(1, 100):
    filename = f"{'test0'}{i // 1000:03d}.{i % 1000:03d}"
    filenames.append(filename)

X_test = []

Xtest = readmap('test.map')
X_test.append(pcr2numpy(Xtest, 2))

# process files for testing data
for i, filename in enumerate(filenames[:99]):
    if os.path.exists(filename): # ensure the file exists
        raster = readmap(filename) # read the raster file
        raster_np = pcr2numpy(raster, 2) # convert to NumPy array
        X_test.append(raster_np)

# convert lists to NumPy arrays and reshape if necessary
X_test = np.array(X_test).reshape(-1, 20, 20, 1)

# save the arrays as .npy files in 'life' folder
np.save('X_test.npy', X_test)