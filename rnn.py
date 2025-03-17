import torch.nn as nn

class RNN(nn.Module):

    def __init__(self,vocab_size,embed_dim=128,hidden_dim=256,num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.rnn = nn.RNN(embed_dim,hidden_dim,batch_first=True)
        self.linear = nn.Linear(hidden_dim,num_classes)
    
    def forward(self,x):

        x = self.embedding(x)
        out,_ = self.rnn(x)
        out = out[:,-1,:]
        out = self.linear(out)
        return out