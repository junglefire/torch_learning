# -*- coding: utf-8 -*- 
#!/usr/bin/env python
import matplotlib.pyplot as plt
import logging as log
import pandas as pd
import numpy as np
import torch
import click

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

@click.command()
@click.option("--cmd", help="show: 显示数据集图形; train: 训练模型", type=str)
@click.option("--dataset", help="数据集存储文件", type=str)
def main(cmd, dataset):
	if cmd == 'show':
		__dataset_show(dataset)
	elif cmd == 'train':
		__model_train()
	else:
		log.error("invalid command!")


# 显示数据集图形
def __dataset_show(dataset):
	log.info("show dataset...")
	rides = pd.read_csv(dataset)
	#我们取出最后一列的前150条记录来进行预测
	counts = rides['cnt'][:150]
	x = np.arange(len(counts))
	y = np.array(counts)
	# 绘制一个图形，展示曲线长的样子
	plt.figure(figsize = (8, 5)) #设定绘图窗口大小
	plt.plot(x, y, 'o-') # 绘制原始数据
	plt.xlabel('X') #更改坐标轴标注
	plt.ylabel('Y') #更改坐标轴标注
	plt.show()

# 训练模型
def __model_train():
	log.info("train model...")
	pass

if __name__ == '__main__':
	main()