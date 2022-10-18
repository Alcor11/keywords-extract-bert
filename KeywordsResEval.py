
# 评估结果

import pandas as pd
from transformers import BertTokenizer, BertConfig
from transformers import BertModel
import torch
from numpy import mean


model_name = 'bert-base-chinese'
# MODEL_PATH = 'D:/exp/bert-base-chinese/'


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

def word_embedding(word,model):     # embedding
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

def write_file(simlist,avg,T,R,F,avgt,avgr,avgf):
    simlist = str(simlist)
    simlist = simlist.replace('[','')
    simlist = simlist.replace(']','')
    avg = str(avg)
    avg = avg.replace('[', '')
    avg = avg.replace(']', '')
    path = 'D:/shiyan/result/bert/assess/BertNormal5Keyresult-11.txt'
    f = open(path,'a', encoding='utf-8')
    f.writelines('sim:'+simlist+'avg:'+avg+'准确率：'+str(T)+'召回率：'+str(R)+'F值：'+str(F)+'当前平均正确率：'+str(avgt)+'当前平均召回率：'+str(avgr)+'当前平均F值：'+str(avgf)+'\n')
    print('write ok')
    return 0

def sim_average(simlist):
    avg = mean(simlist)
    return avg



def main():
    truekeypath = 'data/keyword.txt'
    keypath = 'result/bert/keyres/5Key_Bert_Normal_sim_layer-11.csv'      # 读取模型提取关键词
    simmaxlist = []
    comparekey = [] # 正确关键词
    modulekey = [] # 模型关键词

    correct_list = []
    recall_list = []
    F_list = []


    TP = 0


    f = open(truekeypath, 'r', encoding='utf-8')


    data = pd.read_csv(keypath)
    keys = data['key']
    model = load_bert(model_name, MODEL_PATH)   # 读取模型
    model.eval()


    for line in f.readlines():
        # print(line)
        comparekey.append(line)
        # keywords = line.split(';')
        # print(keywords)
        print('读取正确关键词')
    print('读取正确关键词完成')
    print(comparekey)
    # for i in range(len(comparekey)):
    #     print(comparekey[i])


    for key in keys:    # 模型得到的关键词文件中的每一个文档的关键词（行）
        print(key)
        modulekey.append(key)    # 模型得到关键词转化为list
        print('读取模型关键词')
    print('模型关键词读取完成')
    print(modulekey)


    for i in range(len(modulekey)):
        print(i)
        print(comparekey[i])
        compared = comparekey[i].split(';')
        print(compared)
        NUM = len(compared)
        nowkey = modulekey[i].split()
        print(nowkey)

        # for now in nowkey:
        #     if any(now in value for value in compared):
        #         TP+=1
        #         print('TP:')
        #         print(TP)
        #     else:
        #         FP+=1
        #         print('FP:')
        #         print(FP)
        # T = TP / (TP + FP)  # 准确率
        # print('准确率为：{}'.format(T))
        # R = TP / NUM  # 召回率
        # print('召回率为：{}'.format(R))
        # F = 2 * T * R / (T + R)  # F值
        # print('F值为：{}'.format(F))     # 精准匹配


        for key in compared:
            print('正在比较的当前正确关键词')
            print(key)
            tmp = 0
            key_embedding = word_embedding(key, model)
            for now in nowkey:
                print(now)
                len_key = len(nowkey) # 输出关键词个数
                now_embedding = word_embedding(now,model)
                sim = torch.cosine_similarity(key_embedding, now_embedding, dim=0)
                # if sim == 1:
                #     print('正确')
                #     TP+=1
                #     continue
                # else:
                #     print('错误')
                #     FP+=1
                if sim >=0.65:      # 大于0.65算正确
                    print('true')
                    TP+=1
                    print(TP)
                    break
                sim = sim.float()
                print(sim)
                if sim >= tmp:
                    tmp = sim
                    print(tmp)
            tmp = float(tmp)
            simmaxlist.append(tmp)
            print('当前关键词最大相似度列表')
            print(simmaxlist)

        T = TP / len_key  # 准确率
        correct_list.append(T)
        print('准确率为：{}'.format(T))
        R = TP / NUM  # 召回率
        recall_list.append(R)
        print('召回率为：{}'.format(R))
        F = 2 * T * R / (T + R + 0.0001)  # F值
        F_list.append(F)
        print('F值为：{}'.format(F))

        print('correct_list')
        print(correct_list)
        print('recall_list')
        print(recall_list)
        print('fpoint_list')
        print(F_list)


        correct_avg = sim_average(correct_list)
        print('平均正确率')
        print(correct_avg)
        recall_avg = sim_average(recall_list)
        print('平均召回率')
        print(recall_avg)
        fpoing_avg = sim_average(F_list)
        print('平均F值')
        print(fpoing_avg)

        avg = sim_average(simmaxlist)
        print('平均为{}'.format(avg))
        write_file(simmaxlist,avg,T,R,F,correct_avg,recall_avg,fpoing_avg)
        TP = 0
        NUM = 0
        simmaxlist = []
        print('清空参数')





if __name__ == '__main__':
    main()


#     for now in nowkey:      # 模型得到的关键词中的每一个
#         print(now)          # 当前判断关键词
#         # print(now)
#         now_embedding = word_embedding(now,model)       # 当前关键词嵌入
#         # print(now_embedding)
#         print(keyword)
#         key_embedding = word_embedding(keyword,model)   # 标注关键词嵌入
#         # print(key_embedding)
#         sim = torch.cosine_similarity(now_embedding, key_embedding, dim=0)  # 计算余弦相似度
#         sim = sim.float()
#         if sim >= tmp:
#             tmp = sim
#             print(tmp)
#
#     simmaxlist.append(tmp)
#     print(simmaxlist)
# break
                    # simlist[i].append(sim)

                # i+=1    # 下一个关键词
                # print(simlist)

