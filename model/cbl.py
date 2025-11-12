import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ConceptBottleneckLayer(nn.Module):
    def __init__(self, input_dim, concept_dim, cbl_layer_num=1, bias=True):
        super(ConceptBottleneckLayer, self).__init__()
        assert cbl_layer_num >= 1, "cbl_layer_num must be >= 1"

        layers = [nn.Linear(input_dim, concept_dim, bias=bias)]

        for _ in range(cbl_layer_num - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(concept_dim, concept_dim, bias=bias))

        self.cbl = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbl(x)

        
def save_cbl_model(model, path):
    torch.save(model.state_dict(), path)

def load_cbl_model(path, input_dim, concept_dim, cbl_layer_num=1, bias=True, device='cpu'):
    model = ConceptBottleneckLayer(input_dim, concept_dim, cbl_layer_num=cbl_layer_num, bias=bias)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

