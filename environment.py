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
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,16), dtype=np.int32, minimum=np.concatenate(([0]*10, [-1, -1, 0], [0]*2, [-1])), maximum=np.concatenate(([20]*10, [10, 20, 3], [7]*3)), name='observation')
    self._state = [0]*16
    self._episode_ended = False
    
    self.tetris = Tetris()
    
    self.tetris.start(False)
    
    
    
    self._state[10] = self.tetris.pos.x
    self._state[11] = self.tetris.pos.y
    self._state[13] = self.tetris.currBlock
    self._state[14] = self.tetris.nextPieces[0]


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = [0]*16
    self._episode_ended = False
    
    self.tetris.newGame()
        

    
    
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self._reset()
    
    prevLineClear = self.tetris.lineClears
    
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
    elif action == 5:
        self.tetris.switchHeld()
        
    self.tetris.step()
    
    colHeights = [0]*10
    for row in self.tetris.gameState:
        for i, cell in enumerate(row):
            colHeights[i] += cell.x
            
    for i in range(len(colHeights)):
        self._state[i] = colHeights[i]
    
    self._state[10] = self.tetris.pos.x
    self._state[11] = self.tetris.pos.y
    self._state[12] = self.tetris.rotation // 90
    self._state[13] = self.tetris.currBlock
    self._state[14] = self.tetris.nextPieces[0]
    self._state[15] = self.tetris.heldPiece
    
    
    if self.tetris.dead:
        self._episode_ended = True
    
    
    if self._episode_ended or self.tetris.score > 10000:
      reward = self.tetris.score / 1000
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      reward = 2 if self.tetris.lineClears - prevLineClear == 4 else 0
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
    
    
    
    
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

