import torch
from typing import Optional

@torch.jit.script
class SimpleConfig(object):
    def __init__(self, 
                num_hidden_layers: int=6,
                use_return_dict: bool=False,
                problem_type: str='',
                output_attentions: bool = False,
                output_hidden_states: bool = False,
                
                ):
        self.problem_type = problem_type
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_hidden_layers = num_hidden_layers

        self.use_return_dict = use_return_dict

