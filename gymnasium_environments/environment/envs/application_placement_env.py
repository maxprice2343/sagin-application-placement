import random
import numpy as np
import gymnasium as gym
import pygame as pg
import asyncio
from environment.application_module import Application_Module
from environment.network_node import Network_Node

# These represent the lower and upper bounds on the number of
# activity modules in the environment
NUM_MODULES_LOWER_BOUND = 15
NUM_MODULES_UPPER_BOUND = 15

# These represent the lower and upper bounds on the number of
# network nodes in the environment
NUM_NODES_LOWER_BOUND = 5
NUM_NODES_UPPER_BOUND = 5

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

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.num_modules = random.randint(
            NUM_MODULES_LOWER_BOUND, NUM_MODULES_UPPER_BOUND
        )
        self.num_nodes = random.randint(
            NUM_NODES_LOWER_BOUND, NUM_NODES_UPPER_BOUND
        )
        """self.observation_space = gym.spaces.Dict(
            {
                "first_module" : gym.spaces.Box(
                    low=np.array([0, MODULE_SIZE_LOWER_BOUND,
                                  MODULE_MEMORY_REQUIRED_LOWER_BOUND, MODULE_DATA_SIZE_LOWER_BOUND]),
                    high=np.array([NUM_MODULES_UPPER_BOUND, MODULE_SIZE_UPPER_BOUND,
                                   MODULE_MEMORY_REQUIRED_UPPER_BOUND, MODULE_DATA_SIZE_UPPER_BOUND]),
                    shape=(4,),
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
        )"""
        low=np.concatenate((np.array([0, MODULE_SIZE_LOWER_BOUND, MODULE_MEMORY_REQUIRED_LOWER_BOUND,
            MODULE_DATA_SIZE_LOWER_BOUND])[None, :], np.resize(np.array([0, NODE_SPEED_LOWER_BOUND,
            NODE_BANDWIDTH_LOWER_BOUND, NODE_MEMORY_LOWER_BOUND]), (self.num_nodes, 4))))
        high=np.concatenate((np.array([NUM_MODULES_UPPER_BOUND, MODULE_SIZE_UPPER_BOUND,
            MODULE_MEMORY_REQUIRED_UPPER_BOUND, MODULE_DATA_SIZE_UPPER_BOUND])[None, :],
            np.resize(np.array([NUM_NODES_UPPER_BOUND, NODE_SPEED_UPPER_BOUND, NODE_BANDWIDTH_UPPER_BOUND,
            NODE_MEMORY_UPPER_BOUND]), (self.num_nodes, 4))))
        self.observation_space = gym.spaces.Box(low, high, shape=(self.num_nodes + 1, 4), dtype=np.int64)

        # The action taken by an agent will be placing a specific module on
        # a specific node for processing.
        """self.action_space = gym.spaces.Box(
            low = np.array([NUM_MODULES_LOWER_BOUND, NUM_NODES_LOWER_BOUND]),
            high = np.array([NUM_MODULES_UPPER_BOUND, NUM_NODES_UPPER_BOUND])
        )"""

        self.action_space = gym.spaces.Discrete(1, 0)

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

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}
    
    def _first_module(self):
        """Returns the first module that hasn't started being processed yet"""
        for (_, v) in self.modules.items():
            if not v.processing:
                return v
        return None

    async def step(self, action: int):
        """Moves the environment forward by 1 step (assigning a module to a
        node for processing)"""
        first_module = self._first_module()
        observation = None
        reward = None

        # The environment terminates if all modules have finished being
        # processed
        terminated = True
        for _, v in self.modules.items():
            if not v.done:
                terminated = False
                break
        
        if self.render_mode == "human":
            self._render_frame()

        if first_module is not None:
            module = first_module
            node = self.nodes[action]

            # Adding the module to the node's processing queue
            # asyncio.create_task causes node.add_module to be added to the
            # event loop for processing (at some point, not immediately)
            task = asyncio.create_task(node.add_module(module))

            # If the task is done already, and the result is 0, this means that
            # a module was assigned to a node with insufficient memory for it,
            # which incurs a negative reward
            if task.done():
                if not task.result():
                    result = -1
            # This line allows the event loop to be checked for pending tasks.
            # The effect of these 2 lines is that node.add_module is added to
            # the event loop, but the step function doesn't need to wait for it
            # to finish before continuing.
            await asyncio.sleep(0)

            if self.render_mode == "human":
                self._render_frame()
            
            # Calculates the processing time of the module
            processing_time = module.num_instructions / node.processing_speed
            # Calculates the resource overhead of the current module
            resource_overhead = module.memory_required / node.available_memory
            # Calculates the reward for the processing of this module
            reward = (MAXIMUM_MODULE_PROCESSING_TIME - processing_time) + MAXIMUM_MODULE_PROCESSING_TIME * (1 - resource_overhead)

            observation = self._get_obs()
        else:
            # If there are no modules left, call asyncio.sleep(0) to allow the
            # event loop to be checked for pending tasks.
            await asyncio.sleep(0)
            # No reward is given here because the agent is not taking any actions.
            # TODO: This may cause problems with the training if the agent begins
            # to associate random actions with reward values. May have to make it
            # so agent stops calling step once all modules are assigned, but
            # environment can't terminate yet. How???
            reward = 0

        if self.render_mode == "human":
            self._render_frame()

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
            #self.text_font = pg.font.SysFont("Arial", 20)
            self.window = pg.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()

        # Creates the canvas with a white background to draw on
        canvas = pg.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))

        """module_text_surface = self.text_font.render("Modules:", False, (0,0,0))
        canvas.blit(module_text_surface, (10, 10))

        node_text_surface = self.text_font.render("Nodes:", False, (0,0,0))
        canvas.blit(node_text_surface, (self.window_size - 112, 10))"""

        #TODO: Add labels to illustrations

        #============================ Drawing Modules ==============================#
        module_radius = 10
        module_padding = 5 # Distance between modules
        base_x = 2 * module_radius + module_padding # X-Coordinate of first module
        base_y = base_x # Y-Coordinate of first module
        module_colour = (37, 58, 76) # Dark blue
        column = 0 # Stores the column of the current module
        row = 0 # Stores the row of the current module
        for i, (k, v) in enumerate(self.modules.items()):
            # Iterates through all modules and draws them if they haven't begun
            # processing
            if v.processing == False:
                module_y = base_y + base_y * row
                module_x = base_x + column * (module_radius * 2 + module_padding)
                # If the module has reached the bottom of the window, increment the
                # column
                if module_y + module_radius + module_padding > self.window_size:
                    row = 0
                    column += 1
                    module_y = base_y + base_y * row
                    module_x = base_x + column * (module_radius * 2 + module_padding)
                
                pg.draw.circle(canvas, module_colour, (module_x, module_y), module_radius)
                row += 1
        
        #============================ Drawing Nodes ==============================#
        node_x = self.window_size - 712 # Constant X-Coordinate for all nodes
        node_colour = (226, 131, 89) # Orange
        node_dims = (40, 20) # Length and height (respectively) of nodes
        node_padding = 5 # Distance between nodes
        base_y = node_padding # Y-Coordinate of first node
        for i, (k, v) in enumerate(self.nodes.items()):
            # Iterates through all nodes and draws them
            node_y = base_y + (node_dims[1] + node_padding) * i
            pg.draw.rect(canvas, node_colour, pg.Rect((node_x, node_y), node_dims))
            # Draws the modules that are currently assigned to each node.
            #print(f"Node {k} has {len(v.modules)} modules")
            if len(v.modules) > 0:
                for j in range(len(v.modules)):
                    pg.draw.circle(
                        canvas,
                        module_colour,
                        (node_x + node_dims[0] + (module_radius + node_padding) * ((j + 1) * 2), node_y + module_radius),
                        module_radius
                    )

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
        """Closes all open resources once the environment terminates."""
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
            nodes[i] = [k, v.processing_speed, v.bandwidth, v.available_memory]

        return np.concatenate((modules[None, 0], nodes))

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
            nodes[i] = Network_Node(processing_speed, bandwidth, memory, NUM_MODULES_UPPER_BOUND)
        return nodes