from environment.application_module import Application_Module
from collections import deque
import asyncio

class Network_Node:
    """This class represents a network node. It is capable of processing
    instructions."""
    def __init__(self, processing_speed : int, bandwidth : int, memory : int, max_modules : int):
        # The processing speed of the node in Bytes Per Second
        self.processing_speed = processing_speed
        # The bandwidth available to the node in Bytes Per Second
        self.bandwidth = bandwidth
        # The total memory of the node in Bytes
        self.total_memory = memory
        # The available memory of the node in Bytes
        self.available_memory = memory

        self.modules = deque([], max_modules)

        self.processing = False
    
    async def add_module(self, new_module : Application_Module) -> None:
        """Adds a new module into this node's queue of modules to be processed.
        Contains a loop that continuously processes modules in the queue until the
        queue is empty."""

        if new_module.memory_required <= self.available_memory:
            new_module.start_processing()
            self.modules.appendleft(new_module)
        
        if not self.processing:
            asyncio.create_task(self.process_modules())

    async def process_modules(self): #TODO: Modules are immediately removed from queue, causing them to not be shown on the render
        self.processing = True
        while len(self.modules) != 0:
            module_to_process = self.modules[-1]
            print(module_to_process.num_instructions / self.processing_speed)
            await asyncio.sleep(module_to_process.num_instructions / self.processing_speed)
            module_to_process.finish_processing()
            self.modules.pop()
            self.available_memory -= module_to_process.memory_required
        self.processing = False