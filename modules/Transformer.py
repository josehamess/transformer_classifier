import numpy as np
import torch
import torch.nn as nn
import multiprocess as mp


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, tokens_len, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.tokens_len = tokens_len
        self.num_heads = num_heads
        self.dk = int(self.emb_size / self.num_heads)
        self.scaler = np.sqrt(self.dk)
        self.weights_dict = {}
        for i in range(self.num_heads):
            self.weights_dict[f'head_{i}'] = [  nn.Linear(self.emb_size, int(self.emb_size / self.num_heads)),
                                                nn.Linear(self.emb_size, int(self.emb_size / self.num_heads)),
                                                nn.Linear(self.emb_size, int(self.emb_size / self.num_heads))
                                            ]
        self.W_o = nn.Linear(self.emb_size, self.emb_size)
        self.layernorm = nn.LayerNorm((self.tokens_len, self.emb_size))
        self.dropout = nn.Dropout(0.1)

    
    def project(self, x, weights):

        # performs forward pass of attention head

        x = self.dropout(x)

        return (weights[0](x), weights[1](x), weights[2](x))
    

    def attention(self, projections):

        # performs attention
        # returns x

        attention = torch.matmul(projections[0], torch.permute(projections[1], (0, 2, 1))) / self.scaler
        attention_softmax = torch.exp(attention) / torch.sum(torch.exp(attention))
        x = torch.matmul(attention_softmax, projections[2])

        return x
    

    def multi_head_attention(self, x):

        # performs attention head multiple times followed by linear layer

        args = []
        for i in range(self.num_heads):
            args.append(self.project(x, self.weights_dict[f'head_{i}']))
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(self.attention, args).get()
        pool.close()
        z_concat = torch.zeros((x.shape[0], self.tokens_len, self.emb_size))
        for i in range(len(results)):
            ax1 = int(i * (self.emb_size / self.num_heads))
            ax2 = int((i + 1) * (self.emb_size / self.num_heads))
            z_concat[:, :, ax1:ax2] = results[i]

        return self.dropout(self.layernorm(self.W_o(z_concat) + x))


class TransformerEncoder(MultiHeadAttention):
    def __init__(self, emb_size, tokens_len, num_heads):
        super().__init__(emb_size, tokens_len, num_heads)
        self.emb_size = emb_size
        self.tokens_len = tokens_len
        self.num_heads = num_heads
        self.mha1 = MultiHeadAttention(self.emb_size, self.tokens_len, self.num_heads)
        self.mha2 = MultiHeadAttention(self.emb_size, self.tokens_len, self.num_heads)
        self.mha3 = MultiHeadAttention(self.emb_size, self.tokens_len, self.num_heads)
        self.mha4 = MultiHeadAttention(self.emb_size, self.tokens_len, self.num_heads)
        self.mha5 = MultiHeadAttention(self.emb_size, self.tokens_len, self.num_heads)
        self.mha6 = MultiHeadAttention(self.emb_size, self.tokens_len, self.num_heads)
        self.mha7 = MultiHeadAttention(self.emb_size, self.tokens_len, self.num_heads)
        self.mha8 = MultiHeadAttention(self.emb_size, self.tokens_len, self.num_heads)
        self.fc1 = nn.Linear(self.emb_size, 1)
        self.sigmoid = nn.Sigmoid()
    

    def encoder(self, x):

        x = self.mha1.multi_head_attention(x)
        x = self.mha2.multi_head_attention(x)
        x = self.mha3.multi_head_attention(x)
        x = self.mha4.multi_head_attention(x)
        x = self.mha5.multi_head_attention(x)
        x = self.mha6.multi_head_attention(x)
        x = self.mha7.multi_head_attention(x)
        x = self.mha8.multi_head_attention(x)
        x = x[:, 0, :]
        x = torch.squeeze(self.fc1(x))
        x = self.sigmoid(x)

        return x