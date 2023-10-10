'''
taken from: https://github.com/dmar-bonn/ipp-marl/tree/master
'''

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

class AgentActionSpace:
    def __init__(self, params: Dict):
        self.params = params
        self.spacing = params["experiment"]["constraints"]["spacing"]
        self.min_altitude = params["experiment"]["constraints"]["min_altitude"]
        self.max_altitude = params["experiment"]["constraints"]["max_altitude"]
        self.space_x_dim = 3
        self.space_y_dim = 3
        self.space_z_dim = (self.max_altitude - self.min_altitude) // self.spacing + 1
        # self.num_actions = params["experiment"]["constraints"]["num_actions"]
        self.environment_x_dim = params["environment"]["x_dim"]
        self.environment_y_dim = params["environment"]["y_dim"]
        self.space_dim = np.array(
            [self.space_x_dim, self.space_y_dim, self.space_z_dim]
        )