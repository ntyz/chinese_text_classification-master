# -*- coding: UTF-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import jieba

# 读取文件
def read(read_path):
    with open(read_path, "rb") as fout:
        text = fout.read()
    return text

# 保存至文件
def write(save_path, text):
    with open(save_path, "wb") as fin:
        fin.write(text)

def data_segment(data_path, save_path):

    dirlist = os.listdir(data_path)
    for cate in dirlist:
        cate_path = data_path + cate + "/"
        seg_path = save_path + cate + "/"
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        files = os.listdir(cate_path)
        for texts in files:
            #print texts
            text = read(cate_path + texts)
            text = text.replace("\r\n","")
            text = text.replace(" ","")
            seg_text = jieba.cut(text)
            write(seg_path + texts, " ".join(seg_text))
            
        print "%s 类处理完毕" % cate

    print "分词完毕"


if __name__=="__main__":
    #对训练集进行分词
    data_path = "搜狗实验室内容分类数据集/"  # 未分词分类语料库路径
    save_path = "seg_data/"  # 分词后分类语料库路径
    data_segment(data_path,save_path)

    #对测试集进行分词
    #corpus_path = "datas/"  # 未分词分类语料库路径
    #seg_path = "seg_datatest/"  # 分词后分类语料库路径
    corpus_segment(corpus_path,seg_path)
