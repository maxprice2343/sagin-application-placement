from environment.application_module import Application_Module

class Network_Node:
    """This class represents a network node. It is capable of processing
    instructions."""
    def __init__(self, processing_speed, bandwidth, memory):
        # The processing speed of the node in Bytes Per Second
        self.processing_speed = processing_speed
        # The bandwidth available to the node in Bytes Per Second
        self.bandwidth = bandwidth
        # The memory of the node in Bytes
        self.memory = memory

        self.modules = {}
    
    def add_module(self, module_id, m : Application_Module):
        """Adds a new module into this node's dictionary of modules being
        processed"""
        self.modules[module_id] = m

    def remove_module(self, module_id):
        """Removes a module from this node's dictionary of modules being
        processed"""
        del self.modules[module_id]