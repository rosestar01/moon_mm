import os
import shutil

data_root = r"E:/数据集/flower_dataset/flower_dataset/"
my_data = r"E:/数据集/flower_dataset/my_data/"
cls = os.listdir(data_root)
print(cls)
if not os.path.exists(my_data+"train"):
    os.mkdir(my_data+"train")
if not os.path.exists(my_data+"val"):
    os.mkdir(my_data+"val")
# if not os.path.exists(data_root+"train.txt"):
#     os.mknod(data_root+"train.txt")
# if not os.path.exists(data_root+"val.txt"):
#     os.mknod(data_root+"val.txt")
# if not os.path.exists(data_root+"classes.txt"):
#     os.mknod(data_root+"classes.txt")

with open(my_data+"classes.txt",mode='w') as f1:
    for name in cls:
        f1.write(name)
        f1.write('\n')
f2 = open(my_data+"train.txt",mode='w')
f3 = open(my_data+"val.txt",mode='w')


for t in range(len(cls)):
    i=0
    if not os.path.exists(my_data+'train/'+ cls[t]):
        os.mkdir(my_data + 'train/' + cls[t])
    if not os.path.exists(my_data+ 'val/'+ cls[t]):
        os.mkdir(my_data + 'val/' + cls[t])
    for img in os.listdir(data_root+cls[t]):
        if i % 4 == 0:
            shutil.copy(data_root + cls[t] + '/' + img, my_data+"val/"+cls[t]+'/' + img)
            f3.write(cls[t]+'/'+img+" "+str(t)+'\n')
        else:
            shutil.copy(data_root + cls[t] + '/' + img, my_data + "train/" + cls[t] + '/' + img)
            f2.write(cls[t] + '/' + img + " " + str(t)+'\n')
        i = i + 1