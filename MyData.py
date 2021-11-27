# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

traindata_image_path = './data/traindata/'#png图片的地址
testdata_image_path = './data/traindata/'#png图片的地址

class MyDataset(Dataset):
 def __init__(self, txt_path, train=True, transform = None, target_transform = None):
     
    self.data_path = txt_path
    self.train_flag = train
    fh = open(txt_path, 'r')
    imgs = []
    
    lines = fh.readlines()
    if self.train_flag is True:
            lines = lines[:int(0.7*len(lines))]
    else:
            lines = lines[int(-0.3*len(lines)):]
             
    for line in lines:
        
        line = line.rstrip()
        words = line.split()
        
        if self.train_flag is True:
            words[0] = os.path.join(traindata_image_path,words[0])#拼接图片地址
        else:
             words[0] = os.path.join(testdata_image_path,words[0])#拼接图片地址
             
        imgs.append((words[0], int(words[1])))
        self.imgs = imgs 
        
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.Compose(
            [
                transforms.Resize(size = (32,32)),#尺寸规范
                transforms.ToTensor(),   #转化为tensor
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform     


 def __getitem__(self,  index: int):
    fn, label = self.imgs[index]
    img = Image.open(fn).convert('RGB') 
    if self.transform is not None:
        img = self.transform(img) 
    return img, label

 def __len__(self):
    return len(self.imgs)



'''

 img_path = os.path.join(self.data_path, img_path)
 
    def __getitem__(self, idx: int):
        # img to tensor and label to tensor
        img_path = self.path_list[idx]
        if self.train_flag is True:
            if img_path.split('.')[0] == 'dog' : 
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0]) # split 的是str类型要转换为int
        label = torch.as_tensor(label, dtype=torch.int64) # 必须使用long 类型数据，否则后面训练会报错 expect long
        img_path = os.path.join(self.data_path, img_path)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label
'''