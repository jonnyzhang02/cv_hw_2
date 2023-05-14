'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-05-13 17:38:28
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-05-13 17:43:02
FilePath: \cv_hw_2\test.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn

Copyright (c) 2023 by zhangyang0207@bupt.edu.cn, All Rights Reserved. 
'''
import pickle

# 打开pickle文件
with open('kmeans.pkl', 'rb') as f:
    # 加载pickle文件中的对象
    kmeans = pickle.load(f)

print(kmeans.cluster_centers_.shape)
print(kmeans.cluster_centers_[0].shape)
print(kmeans.labels_)
print(kmeans.inertia_)