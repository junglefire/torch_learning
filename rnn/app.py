# -*- coding: utf-8 -*- 
#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mplt
import logging as log
import numpy as np
import pickle
import torch
import click
import tqdm

from rnn import *


log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

# 全局参数
TRAIN_ERROR_CURVE = "./model/{}_loss.pkl"
TORCH_MODEL_FILE = "./model/{}.model"
NUM_OF_EPOCH = 20
NUM_OF_SAMPLES = 2000
# NUM_OF_EPOCH = 5
# NUM_OF_SAMPLES = 100

@click.command()
@click.option("-c", "--command", help="execute command", type=str)
@click.option("-m", "--model", help="model type", type=str)
def main(command, model):
	if (command=="train"):
		if (model=="rnn"):
			__train_rnn_model(model)
		elif (model=="lstm"):
			__train_lstm_model(model)
		else:
			log.error("invalid train model type")
	elif (command=="eval"):
		if (model=="rnn"):
			__eval_rnn_model(model)
		elif (model=="lstm"):
			__eval_lstm_model(model)
		else:
			log.error("invalid eval model type")
	else:
		log.error("invalid command, abort!")
	pass


#
# 训练模型
def __train_rnn_model(model):
	log.info("generate %d number seq..." % (NUM_OF_SAMPLES))
	(train_set, valid_set) = __gen_context_free_seq(NUM_OF_SAMPLES)
	# 生成一个最简化的RNN，输入size为4，可能值为0,1,2,3，输出size为3，可能值为0,1,2
	rnn = SimpleRNN(input_size = 4, hidden_size = 2, output_size = 3)
	loss_fn = torch.nn.NLLLoss() # 交叉熵损失函数
	optimizer = torch.optim.Adam(rnn.parameters(), lr = 0.001) #Adam优化算法
	# 循环训练
	results = []
	for epoch in range(NUM_OF_EPOCH):
		log.info("epoch-%s..." % (epoch))
		train_loss = 0
		np.random.shuffle(train_set)
		bar = tqdm.tqdm(total=len(train_set), leave=False)
		for i, seq in enumerate(train_set):
			loss = 0
			bar.update(1)
			hidden = rnn.initHidden()  #初始化隐含层神经元
			# 对每一个序列的所有字符进行循环
			for t in range(len(seq) - 1):
				#当前字符作为输入，下一个字符作为标签
				# x尺寸：batch_size = 1, time_steps = 1, data_dimension = 1
				x = torch.LongTensor([seq[t]]).unsqueeze(0)
				# y尺寸：batch_size = 1, data_dimension = 1
				y = torch.LongTensor([seq[t + 1]])
				# output尺寸：batch_size, output_size = 3
				# hidden尺寸：layer_size =1, batch_size=1, hidden_size
				output, hidden = rnn(x ,hidden)
				loss += loss_fn(output, y)
			loss = 1.0*loss/len(seq) # 计算每字符的损失数值
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss 
		# 把结果打印出来
		log.info('epoch-%d, train loss: %.8f' % (epoch, train_loss.data.numpy()/i))
		bar.close()
		# 在校验集上测试
		(valid_loss, errors, show_out) = (0, 0, '')
		# 对每一个valid_set中的字符串做循环
		for i, seq in enumerate(valid_set):
			(loss, outstring, targets, diff) = (0, '', '', 0)
			hidden = rnn.initHidden()
			# 对每一个字符做循环
			for t in range(len(seq) - 1):
				# x尺寸：batch_size = 1, time_steps = 1, data_dimension = 1
				x = torch.tensor([seq[t]], dtype = torch.long).unsqueeze(0)
				# y尺寸：batch_size = 1, data_dimension = 1
				y = torch.tensor([seq[t+1]], dtype = torch.long)
				# output尺寸：batch_size, output_size = 3
				# hidden尺寸：layer_size =1, batch_size=1, hidden_size
				output, hidden = rnn(x, hidden)
				# 改写成这样清楚些：
				# torch.max() returns a namedtuple (values, indices) where values is the maximum value 
				# of each row of the input tensor in the given dimension dim, and indices is the index 
				# location of each maximum value found
				# mm = torch.max(output, 1)[1][0] 	 # 以概率最大的元素作为输出
				mm = torch.max(output, 1).indices[0]
				outstring += str(mm.data.numpy()) 	 # 合成预测的字符串
				targets += str(y.data.numpy()[0]) 	 # 合成目标字符串
				loss += loss_fn(output, y) 		 	 # 计算损失函数
				diff += 1 - mm.eq(y).data.numpy()[0] # 计算模型输出字符串与目标字符串之间差异的字符数量
			loss = 1.0*loss/len(seq)
			valid_loss += loss  # 累积损失函数值
			errors += diff 		# 计算累积错误数
		# 取时间步的最后一个输出
		log.info("last step output: %.4f..." % (output[0][2].data.numpy()))
		tl = train_loss.data.numpy()/len(train_set)
		vl = valid_loss.data.numpy()/len(valid_set)
		rt = 1.0*errors/len(valid_set)
		log.info('epoch-%d, train loss: %.8f, val loss: %.8f, error rate: %.4f' % (epoch, tl, vl, rt))
		results.append([tl, vl, rt])	
	# 存储模型和训练过程的数据
	pickle.dump(results, open(TRAIN_ERROR_CURVE.format(model), "wb"))
	torch.save(rnn, TORCH_MODEL_FILE.format(model))


#
# 评估模型
def __eval_rnn_model(model):
	rnn = torch.load(TORCH_MODEL_FILE.format(model))
	for n in range(20):
	    inputs = [0] * n + [1] * n
	    inputs.insert(0, 3)
	    inputs.append(2)
	    (outstring, targets, diff, hiddens) = ('', '', 0, [])
	    hidden = rnn.initHidden()
	    for t in range(len(inputs) - 1):
	        x = torch.tensor([inputs[t]], dtype = torch.long).unsqueeze(0)
	        # x尺寸：batch_size = 1, time_steps = 1, data_dimension = 1
	        y = torch.tensor([inputs[t + 1]], dtype = torch.long)
	        # y尺寸：batch_size = 1, data_dimension = 1
	        output, hidden = rnn(x, hidden)
	        # output尺寸：batch_size, output_size = 3
	        # hidden尺寸：layer_size =1, batch_size=1, hidden_size
	        hiddens.append(hidden.data.numpy()[0][0])
	        #mm = torch.multinomial(output.view(-1).exp())
	        mm = torch.max(output, 1)[1][0]
	        outstring += str(mm.data.numpy())
	        targets += str(y.data.numpy()[0])
	        diff += 1 - mm.eq(y).data.numpy()[0]
	    # 打印出每一个生成的字符串和目标字符串
	    print('Length:{}\nOutput:{}\nTarget:{}\nDiff:{}\n{}'.format(n, outstring, targets, diff, '='*80))

#
# 生成上下文无关的01序列，以3作为开始、2作为结尾，如300112、30001112等
def __gen_context_free_seq(num_of_samples):
	train_set = []
	valid_set = []
	# 生成的样本数量
	samples = num_of_samples
	# 训练样本中n的最大值
	sz = 10
	# 定义不同n的权重，我们按照10:6:4:3:1:1...来配置字符串生成中的n=1,2,3,4,5,...
	# 按原文的概率，每次预测前两位都是01
	# probability = 1.0 * np.array([10, 6, 4, 3, 1, 1, 1, 1, 1, 1])
	probability = 1.0 * np.array([8, 6, 4, 2, 2, 2, 1, 1, 1, 1])
	# 保证n的最大值为sz
	probability = probability[:sz]
	# 归一化，将权重变成概率
	probability = probability/sum(probability)
	# 开始生成samples这么多个样本
	for m in range(samples):
		# 对于每一个生成的字符串，随机选择一个n，n被选择的权重被记录在probability中
		n = np.random.choice(range(1, sz + 1), p = probability)
		# 生成这个字符串，用list的形式完成记录
		inputs = [0]*n + [1]*n
		# 在最前面插入3表示起始字符，2插入尾端表示终止字符
		inputs.insert(0, 3)
		inputs.append(2)
		train_set.append(inputs) #将生成的字符串加入到train_set训练集中
	# 再生成samples/10的校验样本
	for m in range(samples // 10):
		n = np.random.choice(range(1, sz + 1), p = probability)
		inputs = [0] * n + [1] * n
		inputs.insert(0, 3)
		inputs.append(2)
		valid_set.append(inputs)
	# 再生成若干n超大的校验样本
	for m in range(2):
		n = sz+m
		inputs = [0]*n + [1]*n
		inputs.insert(0, 3)
		inputs.append(2)
		valid_set.append(inputs)
	np.random.shuffle(valid_set)
	return (train_set, valid_set)











# 主程序
if __name__ == '__main__':
	main()