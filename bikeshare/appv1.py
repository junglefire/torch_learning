# -*- coding: utf-8 -*- 
#!/usr/bin/env python
import matplotlib.pyplot as plt
import logging as log
import pandas as pd
import numpy as np
import torch
import click

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

NUM_OF_RECORDS = 50
NUM_OF_EPOCH = 200000

@click.command()
@click.option("--cmd", help="show: 显示数据集图形; train: 训练模型", type=str)
@click.option("--dataset", help="数据集存储文件", type=str)
@click.option("--norm", default=False, help="数据集存储文件", type=bool)
def main(cmd, dataset, norm):
	if cmd == 'show':
		__dataset_show(dataset)
	elif cmd == 'train':
		__model_train(dataset, norm)
	else:
		log.error("invalid command!")

# 显示数据集图形
def __dataset_show(dataset):
	log.info("show dataset...")
	rides = pd.read_csv(dataset)
	#我们取出最后一列的前150条记录来进行预测
	counts = rides['cnt'][:NUM_OF_RECORDS]
	x = np.arange(len(counts))
	y = np.array(counts)
	# 绘制一个图形，展示曲线长的样子
	plt.figure(figsize = (8, 5)) #设定绘图窗口大小
	plt.plot(x, y, 'o-') # 绘制原始数据
	plt.xlabel('X') #更改坐标轴标注
	plt.ylabel('Y') #更改坐标轴标注
	plt.show()

# 训练模型
def __model_train(dataset, norm):
	log.info("train model...")
	# 读取数据集
	rides = pd.read_csv(dataset)
	counts = rides['cnt'][:NUM_OF_RECORDS]
	# 输入变量，1,2,3,...这样的一维数组
	if norm:
		x = torch.FloatTensor(np.arange(len(counts), dtype = float) / len(counts))
	else:
		x = torch.FloatTensor(np.arange(len(counts), dtype = float))
	# 输出变量，它是从数据counts中读取的每一时刻的单车数，共50个数据点的一维数组，作为标准答案
	y = torch.FloatTensor(np.array(counts, dtype = float))
	# 设置隐含层神经元的数量
	sz = 10  
	# 初始化输入层到隐含层的权重矩阵，它的尺寸是(1,10)
	weights = torch.randn((1, sz), requires_grad = True)
	# 初始化隐含层节点的偏置向量，它是尺寸为10的一维向量
	biases = torch.randn((sz), requires_grad = True)
	# 初始化从隐含层到输出层的权重矩阵，它的尺寸是(10,1)
	weights2 = torch.randn((sz, 1), requires_grad = True)
	# 设置学习率
	learning_rate = 0.001 
	losses = []
	x = x.view(NUM_OF_RECORDS, -1)
	y = y.view(NUM_OF_RECORDS, -1)
	for i in range(NUM_OF_EPOCH):
		# 从输入层到隐含层的计算
		hidden = x * weights + biases
	 	# 此时，hidden变量的尺寸是：(50,10)，即50个数据点，10个隐含层神经元
	 	# 将sigmoid函数作用在隐含层的每一个神经元上
		hidden = torch.sigmoid(hidden)
		# 隐含层输出到输出层，计算得到最终预测
		predictions = hidden.mm(weights2)
		# 此时，predictions的尺寸为：(50,1)，即50个数据点的预测数值
		# 通过与数据中的标准答案y做比较，计算均方误差
		loss = torch.mean((predictions - y) ** 2)
		# 此时，loss为一个标量，即一个数
		losses.append(loss.data.numpy())
		if i % 10000 == 0: #每隔10000个周期打印一下损失函数数值
			print('loss:', loss)
		# 接下来开始梯度下降算法，将误差反向传播
		loss.backward() 
		# 利用上一步计算中得到的weights，biases等梯度信息更新weights或biases的数值
		weights.data.add_(-learning_rate * weights.grad.data)
		biases.data.add_(-learning_rate * biases.grad.data)
		weights2.data.add_(-learning_rate * weights2.grad.data)
		# 清空所有变量的梯度值
		weights.grad.data.zero_()
		biases.grad.data.zero_()
		weights2.grad.data.zero_()
	# 绘图
	# plt.plot(losses)
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# 绘制预测结果和原始数据的比较
	x_data = x.data.numpy()  #获得x包裹的数据
	plt.figure(figsize = (10, 7))  #设定绘图窗口大小
	xplot, = plt.plot(x_data, y.data.numpy(), 'o')  #绘制原始数据
	yplot, = plt.plot(x_data, predictions.data.numpy())  #绘制拟合数据
	plt.xlabel('X')  #更改坐标轴标注
	plt.ylabel('Y')  #更改坐标轴标注
	plt.legend([xplot, yplot],['Data', 'Prediction under 1000000 epochs'])  #绘制图例
	plt.show()

if __name__ == '__main__':
	main()