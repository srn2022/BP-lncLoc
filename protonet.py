import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cosine_similarity, euclidean_dist_similarity,manhattan_distance
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim):
        super(BiLSTM_Attention, self).__init__()
        self.lstm = nn.LSTM(input_dim, 5, 4, bidirectional=True)
        self.out = nn.Linear(5 * 2, 4096)

    def attention_net(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, X):
        ''':param X: [batch_size, seq_len]'''
        X = X.view(len(X), 1, -1)  # 把原有2维度[a,b]改为3维[a,1,b]
        output, (final_hidden_state, final_cell_state) = self.lstm(X)
        output = output.transpose(0, 1) #output : [batch_size, seq_len, n_hidden * num_directions(=2)]

        # Apply attention mechanism
        context, attn_weights = self.attention_net(output, final_hidden_state)

        # Apply attention to each time step
        attended_output = output * attn_weights.unsqueeze(2)

        return self.out(attended_output)

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
        x_proto = x_proto.reshape(5, self.k_shot, 4096)  # (n, k, embed_dim)
        x_proto = x_proto.mean(1)  # (n, embed_dim)计算每一行的平均值
        x_q = self.encoder(x_query)  # (n, q, embed_dim)
        x_q = x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)

        sim_result = self.similarity(x_q, x_proto)  # (n*q, n)

        log_p_y = F.log_softmax(sim_result, dim=1)  # (n*q, n)

        return log_p_y  # (n*q, n)


    def similarity(self, a, b):
        sim_type = self.ED
        methods = {'euclidean': euclidean_dist_similarity, 'cosine': cosine_similarity,'manhattan':manhattan_distance}
        assert sim_type in methods.keys(), f'type must be in {methods.keys()}'
        return methods[sim_type](a, b)  # 值越大相似度越高
