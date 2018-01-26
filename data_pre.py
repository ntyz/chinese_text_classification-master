# -*- coding: UTF-8 -*-
import sys
import os
import cPickle as pickle
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
reload(sys)
sys.setdefaultencoding('utf-8')

def readf(path):
    with open(path, "rb") as fout:
        text = fout.read()
    return text

def change(seg_path, save_bunch):

    data_bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    dirlist = os.listdir(seg_path)
    data_bunch.target_name.extend(dirlist)

    for cate in dirlist:
        files = os.listdir(seg_path + cate + "/")  
        for text in files: 
            filepath = seg_path + cate + "/" + text 
            data_bunch.label.append(cate)
            data_bunch.filenames.append(filepath)
            data_bunch.contents.append(readf(filepath))  # 读取文件内容
            
        print "%s cate done" % cate

    # 将bunch存储到路径中
    with open(save_bunch, "wb") as fobj:
        pickle.dump(data_bunch, fobj)
    print "文本对象构建完毕"
    

if __name__ == "__main__":
    save_bunch = "wordbag/data_set.dat"  # Bunch存储路径
    seg_path = "seg_data/"  # 分词后分类语料库路径
    change(seg_path, save_bunch)

