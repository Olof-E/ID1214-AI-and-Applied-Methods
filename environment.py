from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tetris import Tetris
import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
# os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# Actions
# position and rotation is encoded as a number between 0 - 39
# where each number represent a specific column and a certain rotation for that column
# Switching held blocks is encoded as action 40

# Observations
# 0    |  total height
# 1    |  last piece height
# 2    |  bumpiness
# 3    |  row transitions
# 4    |  column transitions
# 5    |  amount of holes
# 6    |  cumulative wells
# 7    |  eroded cells
# 8    |  lines cleared
# 9    |  row holes
# 10   |  current block
# 11   |  next block
# 12   |  held block
# 13   |  switch held block available


class TetrisEnvironment(py_environment.PyEnvironment):
  def __init__(self, slowMode=False):      
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=40, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(14,), dtype=np.float32, minimum=np.concatenate(([0.0], [-1.0], [0.0]*8, [-1.0]*3, [0.0])), maximum=np.concatenate(([2.0], [2.0], [2.0]*3, [2.0]*2, [1.0], [4.0], [1.0], [7.0]*3, [1.0])), name='observation')
    
    self._state = [0.0]*14
    self._episode_ended = False
    
    self.tetris = Tetris(slowMode)
    
    self.tetris.start(False)
    

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = [0.0]*14
    self._episode_ended = False
    
    self.tetris.newGame()
    
    return ts.restart(np.array(self._state, dtype=np.float32))

  def _step(self, action):

    if self._episode_ended: 
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self._reset()
    
    prevLineClear = self.tetris.lineClears
    prevBlock = self.tetris.currBlock
    
    
    # if action == 0:
    #     self.tetris.move(-1)
    # elif action == 1:
    #     self.tetris.move(1)
    # elif action == 2:
    #     self.tetris.rotate(-1)
    # elif action == 3:
    #     self.tetris.rotate(1)
    # elif action == 4:
    #     self.tetris.quickDrop()
    # elif action == 5 and self.tetris.heldAvailable:
    #     self.tetris.switchHeld()
    if action != 40:
      self.tetris.rotation = (action % 4) * 90
      
      while(action // 4 != self.tetris.pos.x):
        prevPos = self.tetris.pos.x
        
        self.tetris.move(int((action // 4 - self.tetris.pos.x)/abs(action // 4 - self.tetris.pos.x)))
        if(prevPos == self.tetris.pos.x):
          break
        
      self._state[1] = self.tetris.quickDrop() / 20
    elif self.tetris.heldAvailable:
      self.tetris.switchHeld()
      
    self.tetris.step()
    
    
    colHeights = [0]*10
    totalHeight = 0
    maxHeight = 0
    minHeight = 99999
    for row in self.tetris.gameState:
      for i, cell in enumerate(row):
        totalHeight += cell.x
    
    for row in self.tetris.gameState:
      for i, cell in enumerate(row):        
        if(cell.x and colHeights[i] == 0):
          colHeights[i] = 20 - i
          
    
    for i in range(len(colHeights)):
      if(colHeights[i] > maxHeight):
        maxHeight = colHeights[i]
      elif(colHeights[i] < minHeight):
        minHeight = colHeights[i]
      
      
    
    self._state[0] = totalHeight / 200
          
    
    totalBump = 0
    for i in range(len(colHeights)):            
      totalBump += (abs(colHeights[i] - colHeights[max(1, i-1)]) + abs(colHeights[i] - colHeights[min(len(colHeights)-2, i+1)]))/2
      
    self._state[2] = totalBump / 100

    rowTotal = 0
    for y in range(self.tetris.playArea.y):
      row_count = 0
      last_empty = False
      for x in range(self.tetris.playArea.x):
        empty = self.tetris.gameState[y][x].x == 0
        if last_empty != empty:
          row_count += 1
          last_empty = empty
        
      if last_empty:
        row_count += 1

      if last_empty and row_count == 2:
          continue

      rowTotal += row_count
    
    self._state[3] = rowTotal / 100

    colTotal = 0
    for x in range(self.tetris.playArea.x):
      col_count = 0
      last_empty = False
      for y in range(self.tetris.playArea.y):
        empty = self.tetris.gameState[y][x].x == 0
        if last_empty != empty:
          col_count += 1
          last_empty = empty
        
      if last_empty:
        col_count += 1

      if last_empty and col_count == 2:
          continue

      colTotal += col_count
    
    self._state[4] = colTotal / 100


    
    totalHoles = 0
    rowHoles = []
    for col in np.transpose(self.tetris.gameState):
      filledCell = False
      for i, cell in enumerate(col):
          if(cell.x and not filledCell):
            filledCell = True
          elif(filledCell and not cell.x):
            totalHoles += 1
            rowHoles.append(i)
    
    self._state[5] = totalHoles / 40
    
    wellDepths = [0]*self.tetris.playArea.x
    for y, row in enumerate(self.tetris.gameState):
      prevEmpty = True
      for x, cell in enumerate(row):
        if cell.x == 0:
          well = False
          nextEmpty = self.tetris.playArea.x > x + 1 >= 0 and self.tetris.gameState[y][x + 1] == 0
          if prevEmpty or nextEmpty:
            well = True
          wellDepths[x] = 0 if well else wellDepths[x] + 1
          prevEmpty = True
        else:
          prevEmpty = False
          
          
    self._state[6] = sum(wellDepths) / 40
    
    self._state[7] = self.tetris.erodedLines / 16
    
    self._state[8] = self.tetris.lineClears - prevLineClear
            
    self._state[9] = len(set(rowHoles)) / 20
    
    self._state[9] = self.tetris.currBlock
    self._state[10] = self.tetris.nextPieces[0]
    self._state[11] = self.tetris.heldPiece
    
    self._state[12] = int(self.tetris.heldAvailable)
        
    if self.tetris.dead:
        self._episode_ended = True



    if self._episode_ended:
      reward = -1
      return ts.termination(np.array(self._state, dtype=np.float32), reward=reward)
    else:
      reward = 0
      if self.tetris.lineClears - prevLineClear > 0:
        print("lines cleared")
        reward += 10 * (self.tetris.lineClears - prevLineClear)**2
        print("reward is: {0}".format(reward))
        
      if self.tetris.currBlock != prevBlock:
        reward += 1
        
        for i, row in enumerate(self.tetris.gameState):
          prevEmpty = True
          maxTotalContinous = -9999
          totalContinous = 0
          for _, cell in enumerate(row):  
            if(cell.x and prevEmpty):
              prevEmpty = False
            elif(cell.x and not prevEmpty):
              totalContinous += 1
            elif(not (cell.x and prevEmpty)):
              if(totalContinous > maxTotalContinous):
                maxTotalContinous = totalContinous
              prevEmpty = True
          
          if(totalContinous >= 4):
            reward += 0.1 + 1 * maxTotalContinous/10 * (i+1)/20
      
      
      return ts.transition(
          np.array(self._state, dtype=np.float32), reward=reward, discount=0.95)

