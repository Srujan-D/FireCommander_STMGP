'''
taken from: https://github.com/dmar-bonn/ipp-marl/tree/master
'''

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from typing import Dict
import numpy as np

from agent.communication_log import CommunicationLog

class Agent():
    def __init__(
            self,
            agent_id: int,
            params: Dict):
        self.agent_id = agent_id
        def kernel_initial(
                σf_initial=1.0,         # covariance amplitude
                ell_initial=1.0,        # length scale
                σn_initial=0.1          # noise level
                ):
            return σf_initial**2 * RBF(length_scale=ell_initial) + WhiteKernel(noise_level=σn_initial)
        
        self.local_fire_gp = GaussianProcessRegressor(kernel=kernel_initial(), n_restarts_optimizer=10)
        self.sensed = False
        
        self.mission_type = self.params["experiment"]["missions"]["type"]
        # self.n_actions = self.params["experiment"]["constraints"]["num_actions"]
        self.v_max = self.params["experiment"]["uav"]["max_v"]
        self.a_max = self.params["experiment"]["uav"]["max_a"]
        self.x_dim = params["environment"]["x_dim"]
        self.y_dim = params["environment"]["y_dim"]
        self.min_altitude = params["experiment"]["constraints"]["min_altitude"]
        self.max_altitude = params["experiment"]["constraints"]["max_altitude"]
        
        self.agent_info = dict()
        self.position = None # [x, y, z]

    def is_in_map(self, position):
        if (
            position[0] >= 0 
            and position[0] <= self.x_dim
            and position[1] >= 0
            and position[1] <= self.y_dim
            and position[2] >= self.min_altitude
            and position[2] <= self.max_altitude
            ):
            return True
        else:
            return False
    
    def communicate(
        self,
        communication_log: CommunicationLog
    ):
        self.agent_info = {
            "position": self.position,
            "local_fire_gp": self.local_fire_gp
            }
        
        global_log = communication_log.store_agent_message(agent_id=self.agent_id, agent_info=self.agent_info)

        return global_log, self.local_fire_gp, self.position
    
    def receive_messages(
        self,
        communication_log: CommunicationLog,
        agent_id: int
    ):
        received_messages = communication_log.get_agent_messages(agent_id=agent_id)

        
    