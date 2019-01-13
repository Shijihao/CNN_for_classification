# -*- coding:utf8 -*-
#!/usr/bin/env python


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(123456)


class CNN(nn.Module):
    """
    Implementation of CNN Concatenation for sentiment classification task
    """

    def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        super(CNN, self).__init__()

        self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
                                embedding_dim=embeddings.size(1),
                                padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.sen_len = max_len
        filter = hidden_dim
        self.output = nn.Linear(2 * filter, output_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, filter, (K, input_dim)) for K in (2, 3)])

    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        """

        :param sen_batch: (batch, sen_length), tensor for sentence sequence
        :param sen_lengths:
        :param sen_mask_matrix:
        :return:
        """

        ''' Embedding Layer | Padding | Sequence_length 40'''
        sen_batch = self.emb(sen_batch)

        batch_size = len(sen_batch)
        conv_batch = sen_batch.view(batch_size, -1, self.input_dim)

        """ CNN Computation"""
        # [batch_size, 1, seq_len, emb]
        conv_batch = conv_batch.unsqueeze(1)
        # [batch_size, hidden_dim, *]
        conv_represent = [conv(conv_batch).squeeze(3) for conv in self.convs]
        pool_represent = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_represent]
        # concated: [batch_size ,hidden_dim*len(Kl)]
        concated = torch.cat(pool_represent, 1)
        representation = concated
        out = self.output(representation)
        out_prob = F.softmax(out.view(batch_size, -1))

        return out_prob