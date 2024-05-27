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