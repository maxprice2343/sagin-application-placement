class Application_Module:
    """This class represents an activity module. It consists of instructions
    that require processing. """
    def __init__(self, num_instructions, memory_required, data_size):
        # The number of instructions in the module
        self.num_instructions = num_instructions
        # The memory required by the module in order to execute, in Bytes
        self.memory_required = memory_required
        # The amount of data required by the node as input, in Bytes
        self.data_size = data_size