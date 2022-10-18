
# Kmeans聚类层获得初选关键词
import os
from sklearn.cluster import KMeans
import pandas as pd
import math
import numpy as np
import jieba.analyse as jieba
import re

from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertConfig
import torch

MODEL_NAME = 'bert-base-chinese'
MODEL_PATH = 'D:/shiyan/bert-base-chinese/'

def preprocess(doc):
    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    comp.sub('',doc)
    return comp.sub('', doc)

def doc_embedding(doc,model):   # embedding
    doc = preprocess(doc)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # document tokenizer
    tokenize_doc = tokenizer.tokenize(doc)
    token_ids = tokenizer.convert_tokens_to_ids(tokenize_doc)

    token_tensor = torch.tensor([token_ids])
    outputs = model(token_tensor)
    if model.config.output_hidden_states:
        hidden_states = outputs[2]
        # last_layer = outputs[-1]
        second_to_last_layer = hidden_states[-2]
        token_vecs = second_to_last_layer[0]
        print(token_vecs.shape)
        # Calculate the average of all input token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding



def textrank_score(doc,stopkey):
    l = []
    pos = ['n', 'nz', 'vn', 'l', 'a', 'd', 'ad', 'ag', 'an', 'ns']  # 定义选取的词性

    seg = jieba.textrank(doc, topK=20, withWeight=True, allowPOS=pos)

    for i in seg:
        if i.word not in l and i.word not in stopkey and i.flag in pos:  # 去重 + 去停用词 + 词性筛选
            # print i.word
            l.append(i.word)
            print(l)

    return l

# 对词向量采用K-means聚类抽取TopK关键词
def getkeywords_kmeans(data,topK):
    words = data["word"] # 词汇
    vecs = data.iloc[:,1:] # 向量表示
    # vecs = data.ix[:, 'total_price']
    kmeans = KMeans(n_clusters=1,random_state=10).fit(vecs)
    labels = kmeans.labels_ #类别结果标签
    labels = pd.DataFrame(labels,columns=['label'])
    new_df = pd.concat([labels,vecs],axis=1)
    df_count_type = new_df.groupby('label').size() #各类别统计个数
    # print df_count_type
    vec_center = kmeans.cluster_centers_ #聚类中心

    # 计算距离（相似性） 采用欧几里得距离（欧式距离）
    distances = []
    vec_words = np.array(vecs) # 候选关键词向量，dataFrame转array
    vec_center = vec_center[0] # 第一个类别聚类中心,本例只有一个类别
    length = len(vec_center) # 向量维度
    for index in range(len(vec_words)): # 候选关键词个数
        cur_wordvec = vec_words[index] # 当前词语的词向量
        dis = 0 # 向量距离
        for index2 in range(length):
            dis += (vec_center[index2]-cur_wordvec[index2])*(vec_center[index2]-cur_wordvec[index2])
        dis = math.sqrt(dis)
        distances.append(dis)
    distances = pd.DataFrame(distances,columns=['dis'])

    res = pd.concat([words, labels ,distances], axis=1) # 拼接词语与其对应中心点的距离
    res = res.sort_values(by="dis",ascending = True) # 按照距离大小进行升序排序


    # 将用于聚类的数据的特征维度降到2维
    pca = PCA(n_components=2)
    new_pca = pd.DataFrame(pca.fit_transform(new_df))
    print(new_pca)
    # 中间向量可视化
    # d = new_pca[new_df['label'] == 0]
    # plt.plot(d[0],d[1],'r.')
    # d = new_pca[new_df['label'] == 1]
    # plt.plot(d[0], d[1], 'go')
    # d = new_pca[new_df['label'] == 2]
    # plt.plot(d[0], d[1], 'b*')
    # plt.gcf().savefig('kmeans.png')
    # plt.show()

    # 抽取排名前topK个词语作为文本关键词
    wordlist = np.array(result['word']) # 选择词汇列并转成数组格式
    print(wordlist)
    if len(wordlist) > topK:
        word_split = [wordlist[x] for x in range(0,topK)] # 抽取前topK个词汇
        print(word_split)
        word_split = " ".join(word_split)
        return word_split
    else:
        word_split = [wordlist[x] for x in range(0, len(wordlist))]  # 抽取前wordlist个词汇
        print(word_split)
        word_split = " ".join(word_split)
        return word_split



def main():
    # 读取数据集
    dataFile = 'abstractdata.csv'
    LayerList = [-3]
    articleData = pd.read_csv(dataFile)
    ids, titles, keys = [], [], []

    for layer in LayerList:
        rootdir = "result/vecsbertlayer" + str(layer) # 词向量文件根目录
        fileList = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(len(fileList)):
            filename = fileList[i]
            path = os.path.join(rootdir,filename)
            if os.path.isfile(path):
                data = pd.read_csv(path, encoding='utf-8') # 读取词向量文件数据
                # print(data)
                artile_keys = getkeywords_kmeans(data,12) # 聚类得到当前文件的关键词
                # 根据文件名获得文章id以及标题
                (shortname, extension) = os.path.splitext(filename) # 得到文件名和文件扩展名
                t = shortname.split("_")
                article_id = int(t[len(t)-1]) # 获得文章id
                artile_tit = articleData[articleData.id==article_id]['title'] # 获得文章标题
                print(list(artile_tit))
                if (list(artile_tit) == []):
                    artile_tit = ''
                else:
                    artile_tit = list(artile_tit)[0] # series转成字符串
                ids.append(article_id)
                titles.append(artile_tit)
                keys.append(artile_keys)
        # 所有结果写入文件
        result = pd.DataFrame({"id": ids, "title": titles, "key": keys}, columns=['id', 'title', 'key'])
        result = result.sort_values(by="id",ascending=True) # 排序
        result.to_csv("result/bert/keys_bertLayer"+ str(layer) + ".csv", index=False)

if __name__ == '__main__':
    main()