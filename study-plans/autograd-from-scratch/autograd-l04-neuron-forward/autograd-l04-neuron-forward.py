import torch

def neuron_forward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> tuple:
    """
    Returns a tuple of scalar tensors: preactivation and tanh output.
    """
    preactivation = torch.sum(inputs * weights) + bias
    output = torch.tanh(preactivation)
    return (preactivation, output)
        
    
    
    