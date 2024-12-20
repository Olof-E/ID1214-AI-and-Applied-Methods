import os

from tetris import Tetris
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
# 16      |  score
# 17      |  died


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
        shape=(18,), dtype=np.int32, minimum=0, name='observation')
    self._state = [0*18]
    self._episode_ended = False
    
    self.tetris = Tetris()
    self.tetris.start()
    
    self._state[10] = self.tetris.pos.x
    self._state[11] = self.tetris.pos.y
    self._state[13] = self.tetris.currBlock
    self._state[14] = self.tetris.nextPieces[0]


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = [0*15]
    self._episode_ended = False
    
    # TODO Reset Tetris board manually (Probably just call init again or something)
    
    self._state[10] = self.tetris.pos.x
    self._state[11] = self.tetris.pos.y
    self._state[13] = self.tetris.currBlock
    self._state[14] = self.tetris.nextPieces[0]
    
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    
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
        pass
    
    self._state[10] = self.tetris.pos.x
    self._state[11] = self.tetris.pos.y
    self._state[13] = self.tetris.currBlock
    self._state[14] = self.tetris.nextPieces[0]
    
    
    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
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