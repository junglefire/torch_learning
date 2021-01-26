# -*- coding: utf-8 -*- 
#!/usr/bin/env python
import torch.nn.functional as F
import torch

class NPLM(torch.nn.Module):
	def __init__(self, vocab_size, embedding_dim, context_size):
		super(NPLM, self).__init__()
		self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)  #嵌入层
		self.linear1 = torch.nn.Linear(context_size * embedding_dim, 128) #线性层
		self.linear2 = torch.nn.Linear(128, vocab_size) #线性层

	def forward(self, inputs):
		# 嵌入运算，嵌入运算在内部分为两步：
		# 1. 将输入的单词编码映射为one hot向量表示
		# 2. 经过一个线性层得到单词的词向量
		# inputs的尺寸为：1*context_size
		embeds = self.embeddings(inputs)
		# embeds的尺寸为: context_size*embedding_dim
		embeds = embeds.view(1, -1)
		# 此时embeds的尺寸为：1*embedding_dim
		# 线性层加ReLU
		out = self.linear1(embeds)
		out = F.relu(out)
		# 此时out的尺寸为1*128
		# 线性层加Softmax
		out = self.linear2(out)
		#此时out的尺寸为：1*vocab_size
		log_probs = F.log_softmax(out, dim = 1)
		return log_probs
	
	def extract(self, inputs):
		embeds = self.embeddings(inputs)
		return embeds