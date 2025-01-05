from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import random
from time import sleep

from tetris import Tetris
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


# Observations
# 0 - 9   |  column height in playing area
# 10 - 11 |  x & y of current piece 
# 12      |  rotation of current piece
# 13      |  current piece
# 14      |  next piece 
# 15      |  held piece



# Actions
# 0  |  Move left
# 1  |  Move Right
# 2  |  Rotate CCW
# 3  |  Rotate CW
# 4  |  Quickdrop piece
# 5  |  Switch held piece



class TetrisEnvironment(py_environment.PyEnvironment):
  def __init__(self):      
    self.step_count = 0

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(22,), dtype=np.int32, minimum=np.concatenate(([0]*5, [-1], [0]*3, [-1]*2, [0], [0]*10)), maximum=np.concatenate(([50]*2, [4], [7]*3, [20]*2, [3], [10, 20], [1], [10]*10)), name='observation')
    # self._observation_spec = array_spec.BoundedArraySpec(
    #     shape=(16,), dtype=np.int32, minimum=np.concatenate(([0]*10, [-1, -1, 0], [0]*2, [-1])), maximum=np.concatenate(([20]*10, [10, 20, 3], [7]*3)), name='observation')
    self._state = [0]*22
    self._episode_ended = False
    
    self.tetris = Tetris()
    
    self.tetris.start(False)
    
    
    
    # self._state[10] = self.tetris.pos.x
    # self._state[11] = self.tetris.pos.y
    self._state[5] = self.tetris.currBlock
    self._state[6] = self.tetris.nextPieces[0]
    self._state[7] = self.tetris.heldPiece

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = [0]*22
    self._episode_ended = False
    
    self.tetris.newGame()
        

    self._state[3] = self.tetris.currBlock
    self._state[4] = self.tetris.nextPieces[0]
    self._state[5] = self.tetris.heldPiece
    
    self._state[8] = self.tetris.rotation // 90    
    self._state[9] = self.tetris.pos.x
    self._state[10] = self.tetris.pos.y
    self._state[11] = int(self.tetris.heldAvailable)
    
    return ts.restart(np.array(self._state, dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self._reset()
    
    prevLineClear = self.tetris.lineClears
    prevBlock = self.tetris.currBlock
    
    if action == 0:
        self.tetris.move(-1)
    elif action == 1:
        self.tetris.move(1)
    elif action == 2:
        self.tetris.rotate(-1)
    elif action == 3:
        self.tetris.rotate(1)
    elif action == 4:
        self.tetris.quickDrop()
    elif action == 5 and self.tetris.heldAvailable:
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
    
    for i in range(len(colHeights)):            
      self._state[i+12] = abs(colHeights[i] - colHeights[max(0, i-1)]) + abs(colHeights[i] - colHeights[min(len(colHeights)-1, i+1)])


    self._state[0] = totalHeight
    
    totalHoles = 0
    for col in np.transpose(self.tetris.gameState):
      filledCell = False
      for i, cell in enumerate(col):
          if(cell.x and not filledCell):
            filledCell = True
          elif(filledCell and not cell.x):
            totalHoles += 1
    
    self._state[1] = totalHoles
    self._state[2] = self.tetris.lineClears - prevLineClear
    self._state[3] = self.tetris.currBlock
    self._state[4] = self.tetris.nextPieces[0]
    self._state[5] = self.tetris.heldPiece
    self._state[6] = maxHeight
    self._state[7] = minHeight

    self._state[8] = self.tetris.rotation // 90        
    self._state[9] = self.tetris.pos.x
    self._state[10] = self.tetris.pos.y
    self._state[11] = int(self.tetris.heldAvailable)
    
            
            
    # for i in range(len(colHeights)):
    #     self._state[i] = colHeights[i]
    
    
    
    # self._state[10] = self.tetris.pos.x
    # self._state[11] = self.tetris.pos.y
    # self._state[12] = self.tetris.rotation // 90
    # self._state[10] = self.tetris.currBlock
    # self._state[11] = self.tetris.nextPieces[0]
    # self._state[12] = self.tetris.heldPiece
    
    
    if self.tetris.dead:
        self._episode_ended = True


  #TODO: Remove reward for just existing(in else statement), look at the agent.py and recalibrate


    #self.step_count = self.step_count + 1
    #print(self.step_count)

  # 4
    if self._episode_ended:
      reward = -500
      if self.tetris.lineClears > 0:
        print(f"steps: ")
        print(f"lines cleared: [{self.tetris.lineClears}]")
        reward += 100 * (self.tetris.lineClears) ** 2
      return ts.termination(np.array(self._state, dtype=np.int32), reward=reward)
    else:
      reward = 0
      reward += 0.01
      return ts.transition(
          np.array(self._state, dtype=np.int32), reward=reward, discount=0.96)



"""
  # Left window 
    if self._episode_ended:
      reward = -400                
      for col in np.transpose(self.tetris.gameState):
        filledCell = False
        for i, cell in enumerate(col):
            if(cell.x and not filledCell):
              filledCell = True
              reward += 0.1
            #elif(filledCell and not cell.x):
            #  reward -= 2
      if self.tetris.lineClears - prevLineClear > 0:
        reward += 100 * (self.tetris.lineClears - prevLineClear)**2
      return ts.termination(np.array(self._state, dtype=np.int32), reward=reward)
    else:
      reward = 0.01
      for col in np.transpose(self.tetris.gameState):
        filledCell = False
        for i, cell in enumerate(col):
            if(cell.x and not filledCell):
              filledCell = True
              reward += 0.01
      return ts.transition(
        np.array(self._state, dtype=np.int32), reward=reward, discount=0.95)
"""




"""
  # 3
    if self._episode_ended:
      reward = -1000
      if self.tetris.lineClears > 0:
        reward += 1000 * (self.tetris.lineClears) ** 2
      return ts.termination(np.array(self._state, dtype=np.int32), reward=reward)
    else:
      reward = 0
      reward += 0.05
      return ts.transition(
          np.array(self._state, dtype=np.int32), reward=reward, discount=0.99)
"""

"""  # 2
    if self._episode_ended:
      reward = -1000
      if self.tetris.lineClears > 0:
        reward += 1000 * (self.tetris.lineClears) ** 2
      return ts.termination(np.array(self._state, dtype=np.int32), reward=reward)
    else:
      reward = 0
      reward += 0.1
      return ts.transition(
          np.array(self._state, dtype=np.int32), reward=reward, discount=0.98)

"""


"""
  # 1
  # Right window
    if self._episode_ended:
      reward = -250
      if self.tetris.lineClears > 0:
        reward += 100 * (self.tetris.lineClears) ** 2
      return ts.termination(np.array(self._state, dtype=np.int32), reward=reward)
    else:
      reward = 0
      reward += 0.1
      return ts.transition(
          np.array(self._state, dtype=np.int32), reward=reward, discount=0.98)
"""










          




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
    
    
    
    
    # # Make sure episodes don't go on forever.
    # if action == 1:
    #   self._episode_ended = True
    # elif action == 0:
    #   new_card = np.random.randint(1, 11)
    #   self._state += new_card
    # else:
    #   raise ValueError('`action` should be 0 or 1.')

    # if self._episode_ended or self._state >= 21:
    #   reward = self._state - 21 if self._state <= 21 else -21
    #   return ts.termination(np.array([self._state], dtype=np.int32), reward)
    # else:
    #   return ts.transition(
    #       np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
    
    
# environment = TetrisEnvironment()

# time_step = environment.reset()
# print('Time step:')
# print(time_step)


# for i in range(600):
#     action = np.array(random.randint(0, 5), dtype=np.int32)
#     sleep(0.015)
#     next_time_step = environment.step(action)
#     print('Next time step:')
#     print(next_time_step)


# environment.tetris.quitGame()

