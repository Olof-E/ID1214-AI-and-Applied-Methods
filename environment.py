from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tetris import Tetris
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
# os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


# Observations
# 0   |  total height
# 1   | last piece height
# 2   | bumpiness
# 3   | row transitions
# 4   | column transitions
# 5   | amount of holes
# 6   | cumulative wells
# 7   | eroded cells
# 8   | lines cleared




class TetrisEnvironment(py_environment.PyEnvironment):
  def __init__(self):      
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=40, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(13,), dtype=np.int32, minimum=np.concatenate(([0], [-1], [0]*7, [-1]*3, [0])), maximum=np.concatenate(([20*10], [20], [100 ]*3, [40]*2, [16], [4], [7]*3, [1])), name='observation')
    
    self._state = [0]*13
    self._episode_ended = False
    
    self.tetris = Tetris()
    
    self.tetris.start(False)
    

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = [0]*13
    self._episode_ended = False
    
    self.tetris.newGame()
    
    return ts.restart(np.array(self._state, dtype=np.int32))

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
        
      self._state[1] = self.tetris.quickDrop()
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
      # self._state[i+8] = colHeights[i]
      if(colHeights[i] > maxHeight):
        maxHeight = colHeights[i]
      elif(colHeights[i] < minHeight):
        minHeight = colHeights[i]
      
      
    # totalHeight /= 10
    
    self._state[0] = totalHeight
          
    
    totalBump = 0
    for i in range(len(colHeights)):            
      totalBump += (abs(colHeights[i] - colHeights[max(1, i-1)]) + abs(colHeights[i] - colHeights[min(len(colHeights)-2, i+1)]))//2
      
    self._state[2] = totalBump

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
    
    self._state[3] = rowTotal

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
    
    self._state[4] = colTotal


    
    totalHoles = 0
    for col in np.transpose(self.tetris.gameState):
      filledCell = False
      for i, cell in enumerate(col):
          if(cell.x and not filledCell):
            filledCell = True
          elif(filledCell and not cell.x):
            totalHoles += 1
    
    self._state[5] = totalHoles
    
    wells = [0]*self.tetris.playArea.x
    for y, row in enumerate(self.tetris.gameState):
      left_empty = True
      for x, cell in enumerate(row):
        if cell.x == 0:
          well = False
          right_empty = self.tetris.playArea.x > x + 1 >= 0 and self.tetris.gameState[y][x + 1] == 0
          if left_empty or right_empty:
            well = True
          wells[x] = 0 if well else wells[x] + 1
          left_empty = True
        else:
          left_empty = False
          
    self._state[6] = sum(wells) 
    
    self._state[7] = self.tetris.erodedLines
    
    self._state[8] = self.tetris.lineClears - prevLineClear
    
    # self._state[9] = minHeight
    # self._state[10] = maxHeight
    
    self._state[9] = self.tetris.currBlock
    self._state[10] = self.tetris.nextPieces[0]
    self._state[11] = self.tetris.heldPiece
    
    self._state[12] = int(self.tetris.heldAvailable)
    
    
    if self.tetris.dead:
        self._episode_ended = True



    if self._episode_ended:
      reward = -1.5
      # for col in np.transpose(self.tetris.gameState):
      #   filledCell = False
      #   for i, cell in enumerate(col):
      #       if(cell.x and not filledCell):
      #         filledCell = True
      #         reward += 0.01
      #       elif(filledCell and not cell.x):
      #         reward -= 0.05
      return ts.termination(np.array(self._state, dtype=np.int32), reward=reward)
    else:
      reward = 0
      if self.tetris.lineClears - prevLineClear > 0:
        print("lines cleared")
        reward += 8 * (self.tetris.lineClears - prevLineClear)**2
        print("reward is: {0}".format(reward))
        
      if self.tetris.currBlock != prevBlock:
        reward += 1
        
        # for row in self.tetris.gameState:
        #   prevEmpty = True
        #   totalContinous = 0
        #   for i, cell in enumerate(row):  
        #     if(cell.x and prevEmpty):
        #       prevEmpty = False
        #     elif(cell.x and not prevEmpty):
        #       totalContinous += 1
        #     elif(not (cell.x and prevEmpty)):
        #       break
          
        #   if(totalContinous >= 6):
        #     reward += 4
        
        # if abs(maxHeight - totalHeight/10) <= 4:
        #   reward += 0.25
        # else:
        #   reward -= 0.45
      
      
      return ts.transition(
          np.array(self._state, dtype=np.int32), reward=reward, discount=0.99)
    
    
    # if self._episode_ended or self.tetris.score > 10000:
    #   reward = self.tetris.score / 1000
    #   return ts.termination(np.array(self._state, dtype=np.int32), reward)
    # else:
    #   reward = 0
    #   if self.tetris.lineClears - prevLineClear == 4:
    #     reward += 2
    #   elif self.tetris.lineClears - prevLineClear > 0:
    #     reward += 0.5
      
    #   for col in np.transpose(self.tetris.gameState):
    #     filledCell = False
    #     for i, cell in enumerate(col):
    #         if(cell.x and not filledCell):
    #           filledCell = True
    #         elif(filledCell and not cell.x):
    #           reward -= 0.25
            
    #   for i in range(len(colHeights)):
    #     if(abs(avgColHeight - colHeights[i]) > 2):
    #       reward -= 0.1
    #     else:
    #       reward += 0.1
          
      
    #   return ts.transition(
    #       np.array(self._state, dtype=np.int32), reward=reward, discount=1.0)

