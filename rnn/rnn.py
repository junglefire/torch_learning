# -*- coding: utf-8 -*- 
#!/usr/bin/env python
import torch.nn.functional as F
import torch

#
# 实现一个简单的RNN模型
class SimpleRNN(torch.nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
		super(SimpleRNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# 一个embedding层
		self.embedding = torch.nn.Embedding(input_size, hidden_size)
		# PyTorch的RNN层，batch_first标志可以让输入的张量的第一个维度表示batch指标
		self.rnn = torch.nn.RNN(hidden_size, hidden_size, num_layers, batch_first = True)
		# 输出的全链接层
		self.fc = torch.nn.Linear(hidden_size, output_size)
		# 最后的logsoftmax层
		self.softmax = torch.nn.LogSoftmax(dim = 1)

	def forward(self, input, hidden):
		# 先进行embedding层的计算，它可以把一个数值先转化为one-hot向量，再把这个向量转化为一个hidden_size维的向量
		# input的尺寸为：batch_size, num_step, data_dim
		x = self.embedding(input)
		# 从输入到隐含层的计算
		# x的尺寸为：batch_size, num_step, hidden_size
		output, hidden = self.rnn(x, hidden)
		# 从输出output中取出最后一个时间步的数值，注意output输出包含了所有时间步的结果,
		# output输出尺寸为：batch_size, num_step, hidden_size
		output = output[:,-1,:]
		# output尺寸为：batch_size, hidden_size
		# 喂入最后一层全链接网络
		output = self.fc(output)
		# output尺寸为：batch_size, output_size
		output = self.softmax(output)
		return output, hidden

	def initHidden(self):
		# 对隐含单元的初始化，尺寸是：layer_size, batch_size, hidden_size
		return torch.zeros(self.num_layers, 1, self.hidden_size)


#
# 一个手动实现的LSTM模型
class SimpleLSTM(torch.nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
		super(SimpleLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# 一个embedding层
		self.embedding = torch.nn.Embedding(input_size, hidden_size)
		# 隐含层内部的相互链接
		self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers, batch_first = True)
		self.fc = torch.nn.Linear(hidden_size, output_size)
		self.softmax = torch.nn.LogSoftmax(dim = 1)

	def forward(self, input, hidden):
		# x的尺寸：batch_size, len_seq, input_size
		x = self.embedding(input)
		# 从输入到隐含层的计算
		# output的尺寸：batch_size, len_seq, hidden_size
		# hidden: (layer_size, batch_size, hidden_size),(layer_size, batch_size,hidden_size)
		output, hidden = self.lstm(x, hidden)
		# output的尺寸：batch_size, hidden_size
		output = output[:,-1,:]
		# output的尺寸：batch_size, output_size
		output = self.fc(output)
		output = self.softmax(output)
		return output, hidden
 
	def initHidden(self):
		hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
		cell = torch.zeros(self.num_layers, 1, self.hidden_size)
		return (hidden, cell)



