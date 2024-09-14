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
        # A queue that stores the modules assigned to each node. This is treated
        # as a FIFO queue, but a deque is used as it supports iteration.
        self.modules = deque([], max_modules)
        # Indicates whet
        self.processing = False
    
    async def add_module(self, new_module : Application_Module) -> None:
        """Adds a new module into this node's queue of modules to be processed."""

        # Checks that the node has the memory available to store this module
        if new_module.memory_required <= self.available_memory:
            # When a new module is added to the queue, the module is marked as
            # being processed
            new_module.start_processing()
            self.modules.appendleft(new_module)

        # If the node is not already processing, calls the function to begin
        # processing modules
        if not self.processing:
            await self.process_modules()

    async def process_modules(self):
        """Begins processing modules in the queue until there are no more."""
        self.processing = True
        while len(self.modules) > 0:
            # Retrieving the module at the front of the queue without removing it
            module_to_process = self.modules[-1]
            # Sleeps for the amount of time required to process the module
            # Asynchronous sleep allows this node to process the module while
            # other nodes can still be assigned and process their own modules.
            await asyncio.sleep(module_to_process.num_instructions / self.processing_speed)
            module_to_process.finish_processing()
            # Removes the finished module from the queue
            self.modules.pop()
            # Frees up the memory occupied by the recently finished module
            self.available_memory -= module_to_process.memory_required
        # If the node finishes all the modules in its queue then it is no longer
        # processing
        self.processing = False