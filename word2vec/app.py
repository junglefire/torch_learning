# -*- coding: utf-8 -*- 
#!/usr/bin/env python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mplt
import logging as log
import numpy as np
import nplm as lm
import pickle
import torch
import click
import jieba
import tqdm
import re

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

# 全局参数
TRAIN_ERROR_CURVE = "./model/nplm_loss.pkl"
TORCH_MODEL_FILE = "./model/nplm.model"
NUM_OF_EPOCH = 100

@click.command()
@click.option("-f", "--filename", help="txt filename", type=str)
@click.option("-c", "--command", help="execute command", type=str)
def main(command, filename):
	if (command=="train"):
		__train_model(filename)
	elif (command=="show"):
		__show_word_vec(filename)
	elif (command=="similary"):
		__find_most_similar(filename)
	else:
		log.error("invalid command, abort!")
	pass

#
# 训练模型
def __train_model(filename):
	(vocab, w2i, i2w, trigrams) = __preprocess(filename)
	# 纪录每一步的损失函数
	losses = [] 
	loss_fn = torch.nn.NLLLoss()
	# 定义NGram模型，向量嵌入维数为10维，N(窗口大小)为2，即三元组的窗口是2
	model = lm.NPLM(len(vocab), 10, 2) 
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001) #使用随机梯度下降算法作为优化器
	# 循环
	for epoch in range(NUM_OF_EPOCH):
		total_loss = torch.Tensor([0])
		bar = tqdm.tqdm(total=len(trigrams), leave=False)
		for idx, (context, target) in enumerate(trigrams):
			bar.update(1)
			# 准备好输入模型的数据，将词汇映射为编码
			context_idxs = [w2i[w][0] for w in context]
			context_var = torch.tensor(context_idxs, dtype = torch.long)
			# 清空梯度：注意PyTorch会在调用backward的时候自动积累梯度信息，故而每隔周期要清空梯度信息一次。
			optimizer.zero_grad()
			# 用神经网络做计算，计算得到输出的每个单词的可能概率对数值
			log_probs = model(context_var)
			loss = loss_fn(log_probs, torch.tensor([w2i[target][0]], dtype = torch.long))
			# 梯度反传
			loss.backward()
			# 对网络进行优化
			optimizer.step()
			# 累加损失函数值
			total_loss += loss.data
		losses.append(total_loss)
		log.info('Epoch#%s, loss is: %.4f ...' % (epoch, total_loss.numpy()[0]))
		bar.close()
	# 保存训练误差和模型
	pickle.dump(losses, open(TRAIN_ERROR_CURVE, "wb"))
	torch.save(model, TORCH_MODEL_FILE)


#
# 显示单词距离
def __show_word_vec(filename):
	log.info("show word vector...")
	(vocab, w2i, i2w, trigrams) = __preprocess(filename)
	model = torch.load(TORCH_MODEL_FILE)
	# 从训练好的模型中提取每个单词的向量
	vec = model.extract(torch.tensor([v[0] for v in w2i.values()], dtype = torch.long))
	vec = vec.data.numpy()
	# 利用PCA算法进行降维
	X_reduced = PCA(n_components=2).fit_transform(vec)
	# 绘制所有单词向量的二维空间投影
	fig = plt.figure(figsize = (20, 10))
	ax = fig.gca()
	ax.set_facecolor('white')
	ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.4, color = 'black')
	# 绘制几个特殊单词的向量
	words = ['智子', '地球', '三体', '质子', '科学', '世界', '文明', '太空', '加速器', '平面', '宇宙', '信息']
	# 设置中文字体，否则无法在图形上显示中文
	zhfont1 = mplt.font_manager.FontProperties(fname='./华文仿宋.ttf', size=15)
	for w in words:
		if w in w2i:
			ind = w2i[w][0]
			xy = X_reduced[ind]
			plt.plot(xy[0], xy[1], '.', alpha =1, color = 'red')
			plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'black')
	plt.show()

	
# 在所有的词向量中寻找到与目标词（word）相近的向量，并按相似度进行排列
def __find_most_similar(filename, word='智子'):
	(vocab, w2i, i2w, trigrams) = __preprocess(filename)
	model = torch.load(TORCH_MODEL_FILE)
	# 从训练好的模型中提取每个单词的向量
	vectors = model.extract(torch.tensor([v[0] for v in w2i.values()], dtype = torch.long))
	vectors = vectors.data.numpy()
	vector = vectors[w2i[word][0]]
	simi = [[__cos_similarity(vector, vectors[num]), key] for num, key in enumerate(w2i.keys())]
	sort = sorted(simi)[::-1]
	words = [i[1] for i in sort]
	print(words[:5])

#
# 定义计算cosine相似度的函数
def __cos_similarity(vec1, vec2):
	norm1 = np.linalg.norm(vec1)
	norm2 = np.linalg.norm(vec2)
	norm = norm1 * norm2
	dot = np.dot(vec1, vec2)
	result = dot/norm if norm > 0 else 0
	return result


#
# 预处理：分词、n-gram
def __preprocess(filename):
	log.info("load txt file: %s..." % (filename))
	with open(filename, 'rt') as f:
		text = f.read()
	# 分词
	temp = jieba.lcut(text)
	words = []
	for i in temp:
		#过滤掉所有的标点符号
		i = re.sub(r"[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", i)
		if len(i) > 0:
			words.append(i)
	# 生成词汇表
	vocab = set(words)
	log.info("vocabular size: %d..." % (len(vocab)))
	# 构建三元组列表
	trigrams = [([words[i], words[i + 1]], words[i + 2]) for i in range(len(words) - 2)]
	# 构建词典
	#有两个字典，一个根据单词索引其编号(word2idx)，一个根据编号索引单词(idx2word)
	word2idx = {}
	idx2word = {}
	ids = 0
	for w in words:
		cnt = word2idx.get(w, [ids, 0])
		if cnt[1] == 0:
			ids += 1
		cnt[1] += 1
		word2idx[w] = cnt
		idx2word[ids] = w
	return (vocab, word2idx, idx2word, trigrams)

# 主程序
if __name__ == '__main__':
	main()