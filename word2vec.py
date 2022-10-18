#!/usr/bin/python
# coding=utf-8
# 预处理，向量化候选词
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告
import codecs
import pandas as pd
import numpy as np
import jieba
import jieba.posseg
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
import torch

# 返回特征词向量
model_name = 'bert-base-chinese'
MODEL_PATH = 'D:/shiyan/bert-base-chinese/'
path = 'D:/shiyan/module.pth'

def load_bert(model_name,MODEL_PATH):
    # a. 通过词典导入分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # b. 导入配置文件
    model_config = BertConfig.from_pretrained(model_name)
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(MODEL_PATH, config = model_config)
    return bert_model


def getWordVecs(wordList, model, layer):
    name = []
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            # if word in model:  # 模型中存在该词的向量表示
            # name.append(word.encode('utf8'))
            tokenizer = BertTokenizer.from_pretrained(model_name)
            # document tokenizer
            tokenized_doc = tokenizer.tokenize(word)
            token_ids = tokenizer.convert_tokens_to_ids(tokenized_doc)
            token_tensor = torch.tensor([token_ids])
            outputs = model(token_tensor)
            if model.config.output_hidden_states:
                hidden_states = outputs[2]
                # last_layer = outputs[-1]
                second_to_last_layer = hidden_states[layer]
                # 由于只要一个句子，所以尺寸为[1, 10, 768]
                token_vecs = second_to_last_layer[0]
                print(token_vecs.shape)
                # Calculate the average of all input token vectors.
                sentence_embedding = torch.mean(token_vecs, dim=0)
                # print(sentence_embedding)
                vecs.append(sentence_embedding)
                name.append(word)
                # print(vecs)

        except KeyError:
            continue
    a = pd.DataFrame(name, columns=['word'])
    vecs = torch.tensor([item.detach().numpy() for item in vecs])
    b = pd.DataFrame(np.array(vecs, dtype='float'))
    return pd.concat([a, b], axis=1)

# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'vn', 'l', 'a', 'd', 'ad', 'ag', 'an','ns']  # 定义选取的词性
    seg = jieba.posseg.lcut(text)  # 分词
    for i in seg:
        if i.word not in l and i.word not in stopkey and i.flag in pos:  # 去重 + 去停用词 + 词性筛选
            # print i.word
            l.append(i.word)
            print(l)
    return l



# def buildAllWordsVecs(title,abstract, model):     # 生成文档向量列表
#
#     # print(abstract)
#     l_ti = preprocess(title)  # 预处理标题
#     l_ab = preprocess(abstract)  # 预处理摘要
#     # 获取候选关键词的词向量
#     doc = str(l_ti + l_ab)  # 拼接数组元素
#     docvecs = doc_embedding(doc, model)  # 获取文档嵌入表示
#     # print(docvecs)
#     return docvecs



# 根据数据预处理文档存储关键词
def get_Word_preprocessing_result(data, stopkey):
    ids = []
    keys = []
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    for index in range(len(idList)):
        id = idList[index]
        title = titleList[index]
        abstract = abstractList[index]
        # print(abstract)
        l_ti = dataPrepos(title, stopkey)  # 处理标题
        l_ab = dataPrepos(abstract, stopkey)  # 处理摘要
        # 获取候选关键词的词向量
        words = np.append(l_ti, l_ab)  # 拼接数组元素
        words = list(set(words))  # 数组元素去重,得到候选关键词列表
        word_split = " ".join(words)
        ids.append(id)
        keys.append(word_split)
    result = pd.DataFrame({"id": ids, "key": keys}, columns=['id', 'key'])
    result.to_csv("result/keys_BertNormal.csv", index=False)

# 根据数据获取候选关键词词向量
def buildAllWordsVecs(data, stopkey, model):
    layers = [-11]
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    for layer in layers:
        for index in range(len(idList)):
            id = idList[index]
            title = titleList[index]
            abstract = abstractList[index]
            # print(abstract)
            l_ti = dataPrepos(title, stopkey)  # 处理标题
            l_ab = dataPrepos(abstract, stopkey)  # 处理摘要
            # 获取候选关键词的词向量
            words = np.append(l_ti, l_ab)  # 拼接数组元素
            words = list(set(words))  # 数组元素去重,得到候选关键词列表
            wordvecs = getWordVecs(words, model, layer)  # 获取候选关键词的词向量表示
            print(wordvecs)
            # 词向量写入csv文件，每个词400维
            data_vecs = pd.DataFrame(wordvecs)
            data_vecs.to_csv('result/vecsbertlayer' + str(layer) + '/wordvecs_' + str(id) + '.csv', index=False)
            print("document ", id, " well done.")

def main():
    # 读取数据集
    dataFile = 'abstractdata.csv'
    data = pd.read_csv(dataFile)
    print(data)
    # 停用词表
    stopkey = [w.strip() for w in codecs.open('data/stopWord.txt', 'r',encoding='utf-8').readlines()]
    # 词向量模型
    # inp = 'wiki.zh.text.vector'
    # model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    model = load_bert(model_name, MODEL_PATH)

    buildAllWordsVecs(data, stopkey, model)
    # get_Word_preprocessing_result(data,stopkey)

if __name__ == '__main__':
    main()