import numpy as np
from agent import Agent

class World():
    def __init__(self, world_size, agent_list, fire_list):
        self.world_size = world_size
        self.agent_list = agent_list
        self.fire_list = fire_list

        self.global_state = np.zeros((self.world_size, self.world_size))
        