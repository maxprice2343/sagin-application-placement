from ast import Module
from curses import beep
import random
import numpy as np
import gymnasium as gym
import pygame as pg

from gymnasium_environments.environment.application_module import Application_Module

from gymnasium_environments.environment.network_node import Network_Node

# These represent the lower and upper bounds on the number of
# activity modules in the environment
NUM_MODULES_LOWER_BOUND = 100
NUM_MODULES_UPPER_BOUND = 145

# These represent the lower and upper bounds on the number of
# network nodes in the environment
NUM_NODES_LOWER_BOUND = 70
NUM_NODES_UPPER_BOUND = 110

# These represent the lower and upper bounds on the processing speed
# of network nodes in Instructions Per Second (IPS)
NODE_SPEED_LOWER_BOUND = 3000000000
NODE_SPEED_UPPER_BOUND = 7000000000

# These represent the lower and upper bounds on the bandwidth of
# network nodes in Bytes Per Second (BPS)
NODE_BANDWIDTH_LOWER_BOUND = 2000000
NODE_BANDWIDTH_UPPER_BOUND = 8000000

# These represent the lower and upper bounds on the available
# memory of network nodes in Bytes
NODE_MEMORY_LOWER_BOUND = 12000000
NODE_MEMORY_UPPER_BOUND = 15000000

# These represent the lower and upper bounds on the size of activity
# modules in # of Instructions
MODULE_SIZE_LOWER_BOUND = 1500000000
MODULE_SIZE_UPPER_BOUND = 2500000000

# These represent the lower and upper bounds on the required memory
# of acitivy modules in Bytes
MODULE_MEMORY_REQUIRED_LOWER_BOUND = 1000000
MODULE_MEMORY_REQUIRED_UPPER_BOUND = 3000000

# These represent the lower and upper bound on the amount of data
# required for a module to execute in Bytes
MODULE_DATA_SIZE_LOWER_BOUND = 200000
MODULE_DATA_SIZE_UPPER_BOUND = 800000

class ApplicationPlacementEnv(gym.Env):
    """ A Gymnasium environment that represents a SAGIN via
    nodes (network devices). Modules are discrete pieces of an
    application that require processing on a node. This environment
    creates the nodes and modules and allows for modules to be placed
    on nodes for processing. It also enforces some constraints."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        num_modules = random.randint(
            NUM_MODULES_LOWER_BOUND, NUM_MODULES_UPPER_BOUND
        )
        self.modules = self._generate_modules(num_modules)
        num_nodes = random.randint(
            NUM_NODES_LOWER_BOUND, NUM_NODES_UPPER_BOUND
        )
        self.nodes = self._generate_nodes(num_nodes)

        # The observation space is a dictionary containing a list of modules
        # and nodes
        # Modules:
        # -----------------------------------------------------------------------
        # Module 1 Num Instructions, Module 1 Memory Required, Module 1 Data Size
        # Module 2 Num Instructions, Module 2 Memory Required, Module 2 Data Size
        # ....
        # -----------------------------------------------------------------------
        # Nodes:
        # -----------------------------------------------------------------------
        # Node 1 Speed, Node 1 Bandwidth, Node 1 Memory
        # Node 2 Speed, Node 2 Bandwidth, Node 2 Memory
        # ....
        # -----------------------------------------------------------------------
        self.observation_space = gym.spaces.Dict(
            {
                "modules" : gym.spaces.Box(
                    low=np.array([MODULE_SIZE_LOWER_BOUND, MODULE_MEMORY_REQUIRED_LOWER_BOUND,
                         MODULE_DATA_SIZE_LOWER_BOUND]),
                    high=np.array([MODULE_SIZE_UPPER_BOUND, MODULE_MEMORY_REQUIRED_UPPER_BOUND,
                          MODULE_DATA_SIZE_UPPER_BOUND])
                ),
                "nodes" : gym.spaces.Box(
                    low=np.array([NODE_SPEED_LOWER_BOUND, NODE_BANDWIDTH_LOWER_BOUND,
                         NODE_MEMORY_LOWER_BOUND]),
                    high=np.array([NODE_SPEED_UPPER_BOUND, NODE_BANDWIDTH_UPPER_BOUND,
                          NODE_MEMORY_UPPER_BOUND])
                )
            }
        )
        # The action taken by an agent will be placing a specific module on
        # a specific node for processing.
        self.action_space = gym.spaces.Discrete(2)

        # Ensures the provided render mode is None (i.e. no rendering will happen)
        # or it is one of the accepted render methods
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
    
    def _generate_modules(self, num_modules):
        """Generates a list of Application_Module objects each with randomly
        generated properties."""
        modules = {}
        for i in range(num_modules):
            # Generates properties of the new modules randomly between set
            # bounds
            num_instructions = random.randint(
                MODULE_SIZE_LOWER_BOUND, MODULE_DATA_SIZE_UPPER_BOUND
            )
            memory_required = random.randint(
                MODULE_MEMORY_REQUIRED_LOWER_BOUND,
                MODULE_MEMORY_REQUIRED_UPPER_BOUND
            )
            data_size = random.randint(
                MODULE_DATA_SIZE_LOWER_BOUND, MODULE_DATA_SIZE_UPPER_BOUND
            )
            # Creates the new module using the properties and appends it to
            # a dictionary
            modules[i] = Application_Module(num_instructions,
                                            memory_required, data_size)
        return modules
    
    def _generate_nodes(self, num_nodes):
        """Generates a list of Node objects each with randomly generated
        properties."""
        nodes = {}
        for i in range(num_nodes):
            # Generates properties of the new nodes randomly between set
            # bounds
            processing_speed = random.randint(
                NODE_SPEED_LOWER_BOUND, NODE_SPEED_UPPER_BOUND
            )
            bandwidth = random.randint(
                NODE_BANDWIDTH_LOWER_BOUND, NODE_BANDWIDTH_UPPER_BOUND
            )
            memory = random.randint(
                NODE_MEMORY_LOWER_BOUND, NODE_MEMORY_UPPER_BOUND
            )
            # Creates the new node using the properties and appends it to
            # a dictionary
            nodes[i] = Network_Node(processing_speed, bandwidth, memory)
        return nodes