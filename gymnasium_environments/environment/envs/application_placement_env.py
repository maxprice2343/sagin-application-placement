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

# Maximum amount of time a module is afforded for processing (seconds)
MAXIMUM_MODULE_PROCESSING_TIME = 1

class ApplicationPlacementEnv(gym.Env):
    """ A Gymnasium environment that represents a SAGIN via
    nodes (network devices). Modules are discrete pieces of an
    application that require processing on a node. This environment
    creates the nodes and modules and allows for modules to be placed
    on nodes for processing. It also enforces some constraints."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
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
        self.num_modules = random.randint(
            NUM_MODULES_LOWER_BOUND, NUM_MODULES_UPPER_BOUND
        )
        self.num_nodes = random.randint(
            NUM_NODES_LOWER_BOUND, NUM_NODES_UPPER_BOUND
        )
        self.num_modules = 5
        self.num_nodes = 2
        self.observation_space = gym.spaces.Dict(
            {
                "modules" : gym.spaces.Box(
                    low=np.resize(np.array([0, MODULE_SIZE_LOWER_BOUND,
                                  MODULE_MEMORY_REQUIRED_LOWER_BOUND, MODULE_DATA_SIZE_LOWER_BOUND]),
                                  (self.num_modules, 4)),
                    high=np.resize(np.array([NUM_MODULES_UPPER_BOUND, MODULE_SIZE_UPPER_BOUND,
                                   MODULE_MEMORY_REQUIRED_UPPER_BOUND, MODULE_DATA_SIZE_UPPER_BOUND]),
                                   (self.num_modules, 4)),
                    shape=(self.num_modules, 4),
                    dtype=np.int64
                ),
                "nodes" : gym.spaces.Box(
                    low=np.resize(np.array([0, NODE_SPEED_LOWER_BOUND,
                                  NODE_BANDWIDTH_LOWER_BOUND, NODE_MEMORY_LOWER_BOUND]),
                                  (self.num_nodes, 4)),
                    high=np.resize(np.array([NUM_NODES_UPPER_BOUND, NODE_SPEED_UPPER_BOUND,
                                   NODE_BANDWIDTH_UPPER_BOUND, NODE_MEMORY_UPPER_BOUND]),
                                   (self.num_nodes, 4)),
                    shape=(self.num_nodes, 4),
                    dtype=np.int64
                )
            }
        )
        # The action taken by an agent will be placing a specific module on
        # a specific node for processing.
        self.action_space = gym.spaces.Box(
            low = np.array([NUM_MODULES_LOWER_BOUND, NUM_NODES_LOWER_BOUND]),
            high = np.array([NUM_MODULES_UPPER_BOUND, NUM_NODES_UPPER_BOUND])
        )

        self.window_size = 1024

        # Ensures the provided render mode is None (i.e. no rendering will happen)
        # or it is one of the accepted render methods
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """Resets the environment by generating a new set of modules and nodes
        with random characteristics"""
        super().reset(seed=seed)

        self.modules = self._generate_modules(self.num_modules)
        self.nodes = self._generate_nodes(self.num_nodes)

        obs = self._get_obs()
        """print(f"# Modules: {self.num_modules}")
        print(f"# Nodes: {self.num_nodes}")
        print(obs)"""
        return obs, {}
    
    def step(self, action):
        """Moves the environment forward by 1 step (assigning a module to a
        node for processing)"""
        module = self.modules[action[0]]
        node = self.nodes[action[1]]

        # Adding the module to the node's processing list
        node.add_module(action[0], module)
        module.start_processing()
        
        # If any module is not yet finished processing then the environment
        # can't terminate yet
        terminated = True
        for k, v in self.modules.items():
            if not v.done:
                terminated = False
                break
        
        # Calculates the processing time of the module
        processing_time = module.num_instructions / node.processing_speed
        # Calculates the memory already used by other modules deployed on the
        # node
        memory_used = 0
        for k, v in node.modules.items():
            if k != action[0]:
                memory_used += v.memory_required
        # Calculates the resource overhead of the current module
        resource_overhead = module.memory_required / (node.memory - memory_used)
        # Calculates the reward for the processing of this module
        # The first part of the calculation incentivizes reduced processing
        # time (it will be between 0 and MAXIMUM_MODULE_PROCESSING_TIME)
        # The second part incentivizes reduced resource overhead. It uses
        # MAXIMUM_MODULE_PROCESSING_TIME to normalize the value (1 - memory_used)
        # which is between 0 and 1, to make it between 0 and
        # MAXIMUM_MODULE_PROCESSING_TIME
        # This ensures that reducing processing speed and reducing resource
        # overhead are considered equal goals.
        reward = (MAXIMUM_MODULE_PROCESSING_TIME - processing_time) + MAXIMUM_MODULE_PROCESSING_TIME * (1 - memory_used)

        observation = self._get_obs()

        """if self.render_mode == "human":
            self._render_frame()"""

        return observation, reward, terminated, False, {}
    
    def render(self):
        """Returns an rgb array representing the environment."""
        return self._render_frame()

    def _render_frame(self):
        """Renders the environment's state using PyGame."""
        # Initializes pygame, and creates a window and/or clock object if they
        # aren't created already
        if self.window is None and self.render_mode == "human":
            pg.init()
            pg.display.init()
            pg.font.init()
            self.text_font = pg.font.SysFont("Arial", 20)
            self.window = pg.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()

        # Creates the canvas with a white background to draw on
        canvas = pg.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))

        #TODO: Add variables to store window size, etc.

        module_text_surface = self.text_font.render("Modules:", False, (0,0,0))
        canvas.blit(module_text_surface, (10,10))

        node_text_surface = self.text_font.render("Nodes:", False, (0,0,0))
        canvas.blit(node_text_surface, (400,10))

        #TODO: Add labels to illustrations

        for i, (k, v) in enumerate(self.modules.items()):
            if v.processing == False:
                pg.draw.circle(canvas, (37, 58, 76), (25, 40 * (k+1)), 10)
        
        for i, (k, v) in enumerate(self.nodes.items()):
            pg.draw.rect(canvas, (226, 131, 89), pg.Rect((400, 40 * (i+1)), (40, 20)))
            print(self.nodes[k].modules)
            if len(self.nodes[k].modules) > 0:
                for j, (kn, vn) in enumerate(self.nodes[k].modules.items()):
                    pg.draw.circle(canvas, (37, 58, 76), (400 * (j+1) + 40 + 15, 45 * (i+1)), 10)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect()) # type: ignore
            pg.event.pump()
            pg.display.update()
            self.clock.tick(self.metadata["render_fps"]) # type: ignore
        else:
            return np.transpose(
                np.array(pg.surfarray.pixels3d(canvas)), axes=(1,0,2)
            )

    def close(self):
        if self.window is not None:
            pg.display.quit()
            pg.quit()

    def _get_obs(self):
        """Translates the environment's current state into an observation"""
        modules = np.ndarray(shape=(len(self.modules), 4), dtype=int)
        for i, (k, v) in enumerate(self.modules.items()):
            modules[i] = [k, v.num_instructions, v.memory_required, v.data_size]
        nodes = np.ndarray(shape=(len(self.nodes), 4), dtype=int)
        for i, (k, v) in enumerate(self.nodes.items()):
            nodes[i] = [k, v.processing_speed, v.bandwidth, v.memory]

        return {
            "modules": modules,
            "nodes": nodes
        }
    
    def _generate_modules(self, num_modules):
        """Generates a dictionary of Application_Module objects each with randomly
        generated properties."""
        modules = {}
        for i in range(num_modules):
            # Generates properties of the new modules randomly between set
            # bounds
            num_instructions = random.randint(
                MODULE_SIZE_LOWER_BOUND, MODULE_SIZE_UPPER_BOUND
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
        """Generates a dictionary of Node objects each with randomly generated
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