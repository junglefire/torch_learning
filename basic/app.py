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
	pass


if __name__ == '__main__':
    main()