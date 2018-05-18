import torch 
a = torch.Tensor([10,11,12,13,14,15])
index = torch.LongTensor([2,1,0,4,3])
b = torch.index_select(a, 0, index)

print("b", b)

print("unsort", torch.index_select(b, 0, index))
