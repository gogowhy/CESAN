# -*- coding: UTF-8 -*-
import os
save_path = '/home/brain-navigation/bishe_cjh/curriculum_clustering/tests/test-data/input/'
find_path = '/home/brain-navigation/bishe_cjh/Market/bounding_box_train'

f = open(save_path+'dataset.txt', 'w')
path_list=os.listdir(find_path)
path_list.sort() #对读取的路径进行排序
for name in path_list:
    if not name[-3:]=='jpg':
        continue
    ID  = name.split('_')
    src_path = find_path + '/' + name
    label = ID[0]
    context = (src_path, label)
    context_all = ' '.join(context)
    f.write(context_all+'\n')
f.close()
