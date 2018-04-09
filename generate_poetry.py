#-*- coding:utf-8 -*-
import numpy as np
from io import open
import sys
import collections

class Poetry:
    def __init__(self):
        self.filename = "poetry"
        self.poetrys = self.get_poetrys()
        self.poetry_vectors,self.word_to_id,self.id_to_word = self.gen_poetry_vectors()
        self.poetry_vectors_size = len(self.poetry_vectors)
        self._index_in_epoch = 0
    #打开数据集对里面古诗进行处理
    def get_poetrys(self):
        poetrys = list()
        f = open(self.filename,"r", encoding='utf-8')
        #获取古诗内容
        for line in f.readlines():
            #返回古诗的标题和内容，由于不需要使用标题，所以这里用_代替
            _,content = line.strip('\n').strip().split(':')
            #去掉古诗中的空格
            content = content.replace(' ','')
            #过滤含有特殊符号的唐诗
            if(not content or '_' in content or '(' in content or '（' in content or "□" in content
                   or '《' in content or '[' in content or ':' in content or '：'in content or '_' in content):
                continue
            #过滤较长或较短的唐诗
            if len(content) < 5 or len(content) > 79:
                continue
            #用|代替古诗中的标点符号
            content_list = content.replace('，', '|').replace('。', '|').split('|')
            flag = True
            #过滤既非五言也非七言的唐诗
            for sentence in content_list:
                slen = len(sentence)
                if 0 == slen:
                    continue
                if 5 != slen and 7 != slen:
                    flag = False
                    break
            if flag:
                #每首古诗以'['开头、']'结尾
                poetrys.append('[' + content + ']')
        return poetrys

    def gen_poetry_vectors(self):
        words = sorted(set(''.join(self.poetrys) + ' '))
        #数字ID到每个字的映射
        id_to_word = {i: word for i, word in enumerate(words)}
        #每个字到数字ID的映射
        word_to_id = {v: k for k, v in id_to_word.items()}
        to_id = lambda word: word_to_id.get(word)
        #唐诗向量化
        poetry_vectors = [list(map(to_id, poetry)) for poetry in self.poetrys]
        return poetry_vectors,word_to_id,id_to_word

    def next_batch(self,batch_size):
        #取数据，断言在出现错误的时候返回
        assert batch_size < self.poetry_vectors_size
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        #取完一轮数据，打乱唐诗集合，重新取数据
        if self._index_in_epoch > self.poetry_vectors_size:
            np.random.shuffle(self.poetry_vectors)
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        #一个batch的长度
        batches = self.poetry_vectors[start:end]

        x_batch = np.full((batch_size, max(map(len, batches))), self.word_to_id[' '], np.int32)
        for row in range(batch_size):
            x_batch[row,:len(batches[row])] = batches[row]
        y_batch = np.copy(x_batch)
        y_batch[:,:-1] = x_batch[:,1:]
        y_batch[:,-1] = x_batch[:, 0]

        return x_batch,y_batch