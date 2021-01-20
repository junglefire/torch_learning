# -*- coding: utf-8 -*- 
#!/usr/bin/env python
import matplotlib.pyplot as plt
import preprocess as pp
import logging as log
import pandas as pd
from util import *
import numpy as np
import pickle
import torch
import click

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

# 全局参数
TRAIN_ERROR_CURVE = "./model/curve.pkl"
TORCH_MODEL_FILE = "./model/jd.model"

@click.command()
@click.option("-c", "--command", help="执行命令", type=str)
@click.option("-p", "--pos-file", help="正面评论存储文件", type=str)
@click.option("-n", "--neg-file", help="负面评论存储文件", type=str)
def main(command, pos_file, neg_file):
	if (command=="train"):
		__train_model(pos_file, neg_file)
	elif (command=="show"):
		__draw_error_curve()
	elif (command=="test"):
		__test_model(pos_file, neg_file)
	elif (command=="analyze"):
		__analyze_model(pos_file, neg_file)
	else:
		log.error("invalid command, abort!")

#
# 训练模型
def __train_model(pos_file, neg_file):
	log.info("pre-precess comment and split dataset...")
	# 预处理、向量化、分割测试集
	tp = pp.TextProcess(pos_file, neg_file)
	pos_sentences, neg_sentences, dictionary = tp.build_dict(True)
	tp.vectorization()
	(train_data, train_label, val_data, val_label, test_data, test_label) = tp.split()
	log.info("train model...")
	# 输入维度为词典的大小：每一段评论的词袋模型
	model = torch.nn.Sequential(
		torch.nn.Linear(len(dictionary), 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 2),
		torch.nn.LogSoftmax(dim=1),
	)
	log.info("Model Info:\n%s" % (model))
	# 损失函数为交叉熵
	cost = torch.nn.NLLLoss()
	# 优化算法为Adam，可以自动调节学习率
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
	records = []
	#循环10个Epoch
	losses = []
	for epoch in range(10):
		for i, data in enumerate(zip(train_data, train_label)):
			x, y = data
			# 需要将输入的数据进行适当的变形，主要是要多出一个batch_size的维度，也即第一个为1的维度
			x = torch.tensor(x, requires_grad = True, dtype = torch.float).view(1,-1)
			# x的尺寸：batch_size=1, len_dictionary
			# 标签也要加一层外衣以变成1*1的张量
			y = torch.tensor(np.array([y]), dtype = torch.long)
			# y的尺寸：batch_size=1, 1
			# 清空梯度
			optimizer.zero_grad()
			# 模型预测
			predict = model(x)
			# 计算损失函数
			loss = cost(predict, y)
			# 将损失函数数值加入到列表中
			losses.append(loss.data.numpy())
			# 开始进行梯度反传
			loss.backward()
			# 开始对参数进行一步优化
			optimizer.step()
			# 每隔3000步，跑一下校验数据集的数据，输出临时结果
			if i % 3000 == 0:
				val_losses = []
				rights = []
				# 在所有校验数据集上实验
				for j, val in enumerate(zip(val_data, val_label)):
					x, y = val
					x = torch.tensor(x, requires_grad = True, dtype = torch.float).view(1,-1)
					y = torch.tensor(np.array([y]), dtype = torch.long)
					predict = model(x)
					# 调用rightness函数计算准确度
					right = accuracy(predict, y)
					rights.append(right)
					loss = cost(predict, y)
					val_losses.append(loss.data.numpy())
				# 将校验集合上面的平均准确度计算出来
				right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
				log.info('Epoch#{}: train cost={:.2f}, valid cost={:.2f}, accuracy={:.2f}'.format(
					epoch, np.mean(losses), np.mean (val_losses), right_ratio))
				records.append([np.mean(losses), np.mean(val_losses), right_ratio])
	# 测试: 在测试集上分批运行，并计算总的正确率
	vals = [] #记录准确率所用列表
	#对测试数据集进行循环
	for data, target in zip(test_data, test_label):
		data, target =torch.tensor(data, dtype = torch.float).view(1,-1), torch.tensor(np.array([target]), dtype = torch.long)
		output = model(data) #将特征数据输入网络，得到分类的输出
		val = accuracy(output, target) #获得正确样本数以及总样本数
		vals.append(val) #记录结果
	#计算准确率
	rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
	test_accuracy = 1.0 * rights[0].data.numpy()/rights[1]
	log.info("test accuracy: %.4f" % (test_accuracy))
	# 保存训练误差和模型
	pickle.dump(records, open(TRAIN_ERROR_CURVE, "wb"))
	torch.save(model, TORCH_MODEL_FILE)

#
# 训练模型
def __test_model(pos_file, neg_file):
	log.info("pre-precess comment and split dataset...")
	# 预处理、向量化、分割测试集
	tp = pp.TextProcess(pos_file, neg_file)
	pos_sentences, neg_sentences, dictionary = tp.build_dict(True)
	tp.vectorization()
	(train_data, train_label, val_data, val_label, test_data, test_label) = tp.split()
	log.info("test model...")
	model = torch.load(TORCH_MODEL_FILE)
	# 测试: 在测试集上分批运行，并计算总的正确率
	vals = [] #记录准确率所用列表
	#对测试数据集进行循环
	for data, target in zip(test_data, test_label):
		data, target =torch.tensor(data, dtype = torch.float).view(1,-1), torch.tensor(np.array([target]), dtype = torch.long)
		output = model(data) #将特征数据输入网络，得到分类的输出
		val = accuracy(output, target) #获得正确样本数以及总样本数
		vals.append(val) #记录结果
	#计算准确率
	rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
	test_accuracy = 1.0 * rights[0].data.numpy()/rights[1]
	log.info("test accuracy: %.4f" % (test_accuracy))


#
# 分析模型
def __analyze_model(pos_file, neg_file):
	log.info("test model...")
	model = torch.load(TORCH_MODEL_FILE)
	log.info("model detail:\n%s" % model.named_parameters)
	# 绘制第二个全连接层的权重大小，model[2]即提取第2层
	# 网络一共有4层，第0层为线性神经元，第1层为ReLU，第2层为第二层神经元连接，第3层为LogSoftmax
	plt.figure(figsize = (8, 5))
	for i in range(model[2].weight.size()[0]):
		weights = model[2].weight[i].data.numpy()
		plt.plot(weights, 'o-', label = i)
		plt.legend()
		plt.xlabel('Neuron in Hidden Layer')
		plt.ylabel('Weights')
	plt.show()


#
# 绘制误差曲线
def __draw_error_curve():
	records = pickle.load(open(TRAIN_ERROR_CURVE, "rb"))
	a = [i[0] for i in records]
	b = [i[1] for i in records]
	c = [i[2] for i in records]
	plt.plot(a, label = 'Train Loss')
	plt.plot(b, label = 'Valid Loss')
	plt.plot(c, label = 'Valid Accuracy')
	plt.xlabel('Steps')
	plt.ylabel('Loss & Accuracy')
	plt.legend()
	plt.show()

# 主程序
if __name__ == '__main__':
	main()