# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from vizdoom import DoomGame,Button,GameVariable,ScreenFormat,ScreenResolution

from tensorforce.environments import Environment
import numpy as np
class ViZDoom(Environment):
	"""
	ViZDoom Integration: https://github.com/mwydmuch/ViZDoom
	"""

	def __init__(self,config_file):
		"""
		Initialize ViZDoom environment.

		Args:
			config_file: .cfg file path, which defines how a world works and look like (maps)
		"""
		self.game = DoomGame()

		# load configurations from file
		self.game.load_config(config_file)
		self.game.init()

		self.state_shape = self.featurize(self.game.get_state()).shape
		self.num_actions = len(self.game.get_available_buttons())
		

	
	def __str__(self):
		return 'ViZDoom'

	def close(self):
		self.game.close()

	def reset(self):
		self.game.new_episode()
		return self.featurize(self.game.get_state())

	def seed(self, seed):
		if seed is None:
			seed = round(time.time())
		self.game.setSeed(seed)
		return seed

	def featurize(self,state):
		if state is None:
			return None
		H = state.screen_buffer.shape[0]
		W = state.screen_buffer.shape[1]
		_vars=state.game_variables.reshape(-1).astype(np.float32)
		_screen_buf=state.screen_buffer.reshape(-1).astype(np.float32)
		
		if state.depth_buffer is None:
			_depth_buf = np.zeros(H*W*1,dtype=np.float32)
		else:
			_depth_buf=state.depth_buffer.reshape(-1).astype(np.float32)

		if state.labels_buffer is None:
			_labels_buf = np.zeros(H*W*1,dtype=np.float32)
		else:
			_labels_buf=state.labels_buffer.reshape(-1).astype(np.float32)

		if state.automap_buffer is None:
			_automap_buf = np.zeros(H*W*1,dtype=np.float32)			
		else:	
			_automap_buf=state.automap_buffer.reshape(-1).astype(np.float32)
		return np.concatenate(
			(_vars, _screen_buf, _depth_buf, _labels_buf, _automap_buf))

	def execute(self, action):		
		one_hot_enc = [0] * self.num_actions
		one_hot_enc[action] = 1
		reward = self.game.make_action(one_hot_enc)
		next_state = self.featurize(self.game.get_state())
		is_terminal = self.game.is_episode_finished()
		return (next_state,is_terminal,reward)

	@property
	def states(self):
		return dict(shape=self.state_shape, type='float')

	@property
	def actions(self):
		return dict(num_actions=self.num_actions, type='int')
