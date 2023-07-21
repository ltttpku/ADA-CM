import torch, pdb

NUM_PER_CHUNK = 60

def forward_chunks(model, input):
    res = []
    chunked_input = torch.tensor_split(input, input.shape[0] // NUM_PER_CHUNK + 1)
    for chunk in chunked_input:
        output = model(chunk)
        res.append(output)
    return torch.cat(res, dim=0)
    