# -*- coding: utf-8 -*- 
#!/usr/bin/env python
import logging as log
import numpy as np
import collections
import jieba
import re

class TextProcess():
	def __init__(self, pos_comment_file: str, neg_comment_file: str)-> None:
		self.pfile = pos_comment_file
		self.nfile = neg_comment_file
		self.dict = {}
		self.all_words = [] 		# 存储所有的单词
		self.pos_sentences = [] 	# 存储正向的评论
		self.neg_sentences = [] 	# 存储负向的评论
		self.dataset = [] 			# 数据集
		self.labels = [] 			# 标签
		self.sentences = [] 		# 原始句子，调试用

	# 扫描所有的文本，分词并建立词典，分出正向还是负向的评论，is_filter可以过滤是否筛选掉标点符号
	def build_dict(self, is_filter: bool=True)-> (list, list, dict):
		# 处理正面评论
		with open(self.pfile, 'r', encoding='utf-8') as fr:
			for idx, line in enumerate(fr):
				if is_filter:
					line = self.__filter_punc(line)
				words = jieba.lcut(line)
				if len(words) > 0:
					self.all_words += words
					self.pos_sentences.append(words)
		log.info('正面评论文件 `%s` 包含 %d 行，%d 个词.' % (self.pfile, idx+1, len(self.all_words)))
		# 处理负面评论
		# count = len(self.all_words)
		with open(self.nfile, 'r', encoding='utf-8') as fr:
			for idx, line in enumerate(fr):
				if is_filter:
					line = self.__filter_punc(line)
				words = jieba.lcut(line)
				if len(words) > 0:
					self.all_words += words
					self.neg_sentences.append(words)
		log.info('负面评论文件 `%s` 包含 %d 行，%d 个词.' % (self.nfile, idx+1, len(self.all_words)))
		#建立词典，diction的每一项为{w:[id, 单词出现次数]}
		cnt = collections.Counter(self.all_words)
		for word, freq in cnt.items():
			self.dict[word] = [len(self.dict), freq]
		log.info('字典大小：%d' % (len(self.dict)))
		return(self.pos_sentences, self.neg_sentences, self.dict)
	
	# 向量化
	def vectorization(self)-> (list, list):
		# 遍历所有句子，将每一个词映射成编码
		# 处理正向评论
		for sentence in self.pos_sentences:
			new_sentence = []
			for word in sentence:
				if word in self.dict:
					new_sentence.append(self.word2index(word))
				log.debug("word: %s, word index: %s" % (word, self.word2index(word)))
			self.dataset.append(self.__sentence2vec(new_sentence))
			self.labels.append(0) #正标签为0
			self.sentences.append(sentence)
		# 处理负向评论
		for sentence in self.neg_sentences:
			new_sentence = []
			for word in sentence:
				if word in self.dict:
					new_sentence.append(self.word2index(word))
			self.dataset.append(self.__sentence2vec(new_sentence))
			self.labels.append(1) #负标签为1
			self.sentences.append(sentence)
		return self.dataset, self.labels

	# 切分测试集(2/5)、验证集(1/5)和训练集(2/5)
	def split(self)-> (list, list, list, list, list, list):
		# 打乱所有的数据顺序，形成数据集
		# indices为所有数据下标的一个全排列
		indices = np.random.permutation(len(self.dataset))
		# 重新根据打乱的下标生成数据集dataset，标签集labels，以及对应的原始句子sentences
		# [注意] 不要修改原有的数据集
		_dataset 	= [self.dataset[i] for i in indices]
		_labels 	= [self.labels[i] for i in indices]
		_sentences 	= [self.sentences[i] for i in indices]
		# 对整个数据集进行划分，分为：训练集、校准集和测试集，其中校准和测试集合的长度都是整个数据集的10分之一
		test_size 	= len(self.dataset) // 10
		train_data 	= _dataset[2 * test_size :]
		train_label = _labels[2 * test_size :]
		val_data 	= _dataset[: test_size]
		val_label = _labels[: test_size]
		test_data 	= _dataset[test_size : 2 * test_size]
		test_label 	= _labels[test_size : 2 * test_size]
		return (train_data, train_label, val_data, val_label, test_data, test_label)

	# 将文本中的标点符号过滤掉
	def __filter_punc(self, sentence):
		sentence = re.sub(r"[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
		return(sentence)

	# 输入一个句子和相应的词典，得到这个句子的向量化表示
	# 向量的尺寸为词典中词汇的个数，i位置上面的数值为第i个单词在sentence中出现的频率
	def __sentence2vec(self, sentence):
		vector = np.zeros(len(self.dict))
		for word_index in sentence:
			vector[word_index] += 1
		return(1.0*vector/len(sentence))

	# 根据单词返还单词的编码
	def word2index(self, word):
		if word in self.dict:
			value = self.dict[word][0]
		else:
			value = -1
		return(value)

	# 根据编码获得单词
	def index2word(sefl, index):
		for w, v in self.dict.items():
			if v[0] == index:
				return(w)
		return(None)










