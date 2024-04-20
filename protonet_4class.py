import os
import torch.nn as nn
import torch.nn.functional as F
from utils import cosine_similarity, euclidean_dist_similarity
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BiLSTM_Attention(nn.Module):
    def __init__(self,input_dim ):
        super(BiLSTM_Attention, self).__init__()
        # self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.lstm = nn.LSTM(input_dim, 5, 4, bidirectional=True)
        self.out = nn.Linear(5 * 2, 64)

    def forward(self,X):
        ''':param X: [batch_size, seq_len]'''
        X = X.view(len(X), 1, -1)  # 把原有2维度[a,b]改为3维[a,1,b]
        output, (final_hidden_state, final_cell_state) = self.lstm(X)
        output = output.transpose(0, 1) #output : [batch_size, seq_len, n_hidden * num_directions(=2)]

        return self.out(output)# attn_output : [batch_size, num_classes], attention : [batch_size, n_step]

# class Encoder(nn.Module):
#     def __init__(self,input_dim):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(  # 变成二维
#             nn.Linear(input_dim, 1024),  # 线性层
#             nn.ReLU(),  # ReLu的激活函数
#             nn.Linear(1024, 512), nn.ReLU(),
#             nn.Linear(512, 256), nn.ReLU(),
#             nn.Linear(256, 128), nn.ReLU(),
#             nn.Linear(128, 64))
#
#     def forward(self, x):
#         # raw_shape = x.shape
#         # x = x.view(-1, *raw_shape[-3:])
#         x = self.encoder(x)
#         # x = x.view(*raw_shape[:-3], -1)
#         return x


class PrototypicalNetwork(nn.Module):
    def __init__(self, input_dim,k_shot, ED):
        super(PrototypicalNetwork, self).__init__()##子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西。
        self.encoder = BiLSTM_Attention(input_dim)
        self.k_shot = k_shot
        self.ED = ED

    def forward(self, x_support, x_query):
        """
        infer an n-way k-shot task
        :param x_support: (n, k, embed_dim)
        :param x_query: (n, q, embed_dim)
        :return: (q, n)
        """
        x_proto = self.encoder(x_support)  # (n*k, embed_dim)
        # x_proto=np.reshape(x_proto, (5, 10, 128))  # (n, k, embed_dim)
        x_proto = x_proto.reshape(4, self.k_shot, 64)  # (n, k, embed_dim)
        x_proto = x_proto.mean(1)  # (n, embed_dim)计算每一行的平均值
        x_q = self.encoder(x_query)  # (n, q, embed_dim)
        x_q = x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)

        sim_result = self.similarity(x_q, x_proto)  # (n*q, n)

        log_p_y = F.log_softmax(sim_result, dim=1)  # (n*q, n)

        return log_p_y  # (n*q, n)


    def similarity(self, a, b):
        sim_type = self.ED
        methods = {'euclidean': euclidean_dist_similarity, 'cosine': cosine_similarity}
        assert sim_type in methods.keys(), f'type must be in {methods.keys()}'
        return methods[sim_type](a, b)  # 值越大相似度越高