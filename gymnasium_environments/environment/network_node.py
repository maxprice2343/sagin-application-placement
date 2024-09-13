from environment.application_module import Application_Module
import queue
import asyncio

class Network_Node:
    """This class represents a network node. It is capable of processing
    instructions."""
    def __init__(self, processing_speed : int, bandwidth : int, memory : int, max_modules : int):
        # The processing speed of the node in Bytes Per Second
        self.processing_speed = processing_speed
        # The bandwidth available to the node in Bytes Per Second
        self.bandwidth = bandwidth
        # The memory of the node in Bytes
        self.memory = memory

        self.modules = queue.Queue(max_modules)
    
    async def add_module(self, new_module : Application_Module) -> None:
        """Adds a new module into this node's queue of modules to be processed.
        Contains a loop that continuously processes modules in the queue until the
        queue is empty."""

        self.modules.put(new_module)

        while not self.modules.empty():
            module_to_process = self.modules.get()
            module_to_process.start_processing()
            await asyncio.sleep(module_to_process.num_instructions / self.processing_speed)
            module_to_process.finish_processing()