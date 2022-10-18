
# 候选关键词相似度计算

import pandas as pd
import re
from transformers import BertTokenizer, BertConfig
from transformers import BertModel
import torch
import jieba
import jieba.posseg
import jieba.analyse

model_name = 'bert-base-chinese'
MODEL_PATH = 'D:/shiyan/bert-base-chinese/'


def load_bert(model_name,MODEL_PATH):       # 读取模型
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

def preprocess(doc):
    # keep English, digital and Chinese
    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    comp.sub('', doc)
    return comp.sub('', doc)



def doc_embedding(doc, model, layer):  # 使用BERT层，取出BERT倒数第layer层向量
    vecs = []
    # doc = preprocess(doc)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # document tokenizer
    tokenized_doc = tokenizer.tokenize(doc)
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_doc)

    if len(token_ids) <= 512:
        token_ids.extend([0] * (512 - len(token_ids)))
    elif len(token_ids) >= 512:
        token_ids = token_ids[:512] # 修整长度

    print(token_ids)
    token_tensor = torch.tensor([token_ids])
    outputs = model(token_tensor)
    if model.config.output_hidden_states:
        hidden_states = outputs[2]
        # last_layer = outputs[-1]
        second_to_last_layer = hidden_states[layer]
        # 由于只要一个句子，所以尺寸为[1, 10, 768]
        token_vecs = second_to_last_layer[0]
        print(second_to_last_layer)
        print(token_vecs)
        print(token_vecs.shape)
        # Calculate the average of all input token vectors.
        # sentence_embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding = token_vecs[0]
        print("句子embedding")
        print(sentence_embedding)
        return sentence_embedding
        # vecs = torch.tensor([item.detach().numpy() for item in vecs])
        # b = pd.DataFrame(np.array(vecs, dtype='float'))
        # return pd.concat([b], axis=1)


def buildAllWordsVecs(title,abstract, model, layer):     # 生成文档向量列表 [cls]

    # print(abstract)
    l_ti = preprocess(title)  # 预处理标题
    l_ab = preprocess(abstract)  # 预处理摘要
    # 获取候选关键词的词向量
    doc = str('[CLS]' + l_ti + l_ab)  # 拼接数组元素
    print(doc)
    docvecs = doc_embedding(doc, model, layer)  # 获取文档嵌入表示
    # print(docvecs)
    return docvecs

def word_embedding(word,model):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # word tokenizer
    tokenized_doc = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_doc)
    token_tensor = torch.tensor([token_ids])
    outputs = model(token_tensor)
    if model.config.output_hidden_states:
        hidden_states = outputs[2]
        # last_layer = outputs[-1]
        second_to_last_layer = hidden_states[-2]
        # 由于只要一个句子，所以尺寸为[1, 10, 768]
        token_vecs = second_to_last_layer[0]
        print(token_vecs.shape)
        # Calculate the average of all input token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding

def sim_evl(simlist,keylist):   # 返回相似度排名前K的词
    topK = 5
    simil = sorted(simlist,reverse=True)
    print(simil)
    key = []
    for s in simil:
        dex = simlist.index(s)
        key.append(keylist[dex])
        if len(key) == topK:
            return key

# def key_write(keylist):
#     with open('result/w2vfinalkeys.txt','r+',encoding='utf-8')as f:
#         f.writelines()


def keyword_get():
    keypathList = [-11]
    modulekey = [] # 模型关键词
    sim_list = []
    key_list = []
    res_keys = []
    id_list = []
    for keypaths in keypathList:
        keypath = 'result/bert/keys_bertLayer' + str(keypaths) + '.csv'
        data = pd.read_csv(keypath)
        keys = data['key']
        model = load_bert(model_name, MODEL_PATH)   # 读取模型
        model.eval()
        dataFile = 'abstractdata.csv'
        data = pd.read_csv(dataFile)
        idList, titleList, abstractList = data['id'], data['title'], data['abstract']

        for key in keys:    # 模型得到的关键词文件中的每一个文档的关键词（行）
            print(key)
            modulekey.append(key)    # 模型得到关键词转化为list
            print('读取模型关键词')
        print('模型关键词读取完成')
        print(modulekey)

        for index in range(len(idList)):     # 计算文档与词的相似度
            # id = idList[index]
            title = titleList[index]
            abstract = abstractList[index]
            # print(abstract)
            docvecs = buildAllWordsVecs(title,abstract,model,keypaths)

            nowkey = modulekey[index].split()
            print(nowkey)
            for now in nowkey:
                print(now)
                now_embedding = word_embedding(now, model)
                # print(now_embedding)
                sim = torch.cosine_similarity(now_embedding,docvecs,dim=0)
                print('第{}篇文章，词语：{}'.format(index+1,now))
                print(sim)
                sim = float(sim)
                sim_list.append(sim)
                key_list.append(now)
            print('sim_list')
            print(sim_list)
            print('key_list')
            print(key_list)
            key = sim_evl(sim_list,key_list)
            print(key)
            key = " ".join(key)
            print(key)
            id = index + 1
            id_list.append(id)             # id list
            res_keys.append(key)            # keywords list
            sim_list = []
            key_list = []                   # 初始化list

            print(res_keys)
            result = pd.DataFrame({"id":id_list,"key": res_keys}, columns=['id','key'])
            # result = result.sort_values(by="id", ascending=True)  # 排序
            result.to_csv("result/bert/keyres/5Key_Bert_Normal_sim_layer" + str(keypaths) + ".csv", index=False)  # 写入csv


def textrank_keyword_get(doc):
    keywords = jieba.analyse.textrank(doc,topK=5, withWeight=False)
    return keywords



if __name__ == '__main__':
    keyword_get()
    # textrank_keyword_get(doc=)


