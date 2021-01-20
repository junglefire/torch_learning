# -*- coding: utf-8 -*- 
#!/usr/bin/env python
import logging as log
import torch

# 计算预测错误率的函数，其中predictions是模型给出的一组预测结果
# batch_size行num_classes列的矩阵，labels是数据之中的正确答案
def accuracy(predictions, labels):
	# 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    pred = torch.max(predictions.data, 1)[1] 
    # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    rights = pred.eq(labels.data.view_as(pred)).sum() 
    # 返回正确的数量和这一次一共比较了多少元素
    return rights, len(labels) 










