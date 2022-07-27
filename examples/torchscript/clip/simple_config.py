import torch

@torch.jit.script
class SimpleConfig(object):
    def __init__(self, 
                output_attentions: bool=False,
                output_hidden_states: bool=False,
                use_return_dict: bool=False
                ):
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

