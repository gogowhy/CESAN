import os
from shutil import copyfile

download_path = '/home/brain-navigation/bishe_cjh/Market'


if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------

train_path_group = []

train_path_group.append(download_path + '/1')
train_path_group.append(download_path + '/2')
train_path_group.append(download_path + '/3')

train_save_path_group = []
train_save_path_group.append(download_path + '/pytorch/1')
train_save_path_group.append(download_path + '/pytorch/2')
train_save_path_group.append(download_path + '/pytorch/3')


for i in range(3):
    train_path = train_path_group[i]
    train_save_path = train_save_path_group[i]
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)