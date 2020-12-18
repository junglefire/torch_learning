# -*- coding: utf-8 -*- 
#!/usr/bin/env python
import matplotlib.pyplot as plt
import logging as log
import torch
import click

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

@click.command()
@click.option("--cmd", help="show: 显示数据集图形; train: 训练模型", type=str)
def main(cmd):
    if cmd == 'show':
    	__dataset_show()
    elif cmd == 'train':
    	__model_train()
    else:
    	log.error("invalid command!")

# 生成数据集
def __gen_dataset():
	x = torch.linspace(0, 100, 100).type(torch.FloatTensor)
	y = x + torch.randn(100)* 10
	return x, y

# 显示数据集图形
def __dataset_show():
	log.info("show dataset...")
	x, y = __gen_dataset()
	plt.figure(figsize=(8, 6)) #设定绘制窗口大小为10*8 inch
	#绘制数据，由于x和y都是自动微分变量，需要用data获取它们包裹的Tensor，并转成Numpy 
	plt.plot(x.data.numpy(), y.data.numpy(), 'o')
	plt.xlabel('X') #添加X轴的标注
	plt.ylabel('Y') #添加Y轴的标注
	plt.show() #画出图形

# 训练模型
def __model_train():
	log.info("train model...")
	x, y = __gen_dataset()
	# 切换测试集和训练集
	x_train, x_test = x[: -10], x[-10 :]
	y_train, y_test = y[: -10], y[-10 :]
	# 初始化参数
	a = torch.rand(1, requires_grad = True)
	b = torch.rand(1, requires_grad = True)
	# 设置学习率
	learning_rate = 0.0001
	# 模型训练
	for i in range(1000):
    	predictions = a.expand_as(x_train) * x_train + b.expand_as(x_train)
    	# 将所有训练数据代入模型ax+b，计算每个的预测值。这里的x_train和predictions都是（90，1）的张量。
    	# expand_as的作用是将a,b扩充维度到和x_train一致
    	loss = torch.mean((predictions - y_train) ** 2)
    	log.info('loss:', loss)
    	loss.backward() #对损失函数进行梯度反传
    	# 利用上一步计算中得到的梯度信息更新参数中的data数值
    	a.data.add_(- learning_rate * a.grad.data)
    	b.data.add_(- learning_rate * b.grad.data)
    	# 清空存储在变量a、b中的梯度信息，以免在backward的过程中反复不停地累加
    	a.grad.data.zero_() #清空a的梯度数值
    	b.grad.data.zero_() #清空b的梯度数值


if __name__ == '__main__':
    main()