import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import spacy
from torch.nn.utils.rnn import pad_sequence

spacy_eng=spacy.load('en_core_web_sm')

class Vocablary():
    
    def __init__(self):
        
        self.stoi = {'<PAD>': 0,'<STR>':1, '<END>':2,'<UNK>':3}
        self.itos = {}
        for s,i in self.stoi.items():
            self.itos[i]=s
     
    def vocab(self,texts,threshold=5):

        count={}
        idx=4
        for cap in texts:

            for word in spacy_eng.tokenizer(str(cap)):
                word=word.text.lower()
                if (word not in self.stoi) and (word not in count) :
                    count[word] =1
                else:
                    count[word] +=1 
                    
                if count[word] == threshold:
                    self.stoi[word]= idx
                    self.itos[idx] = word
                    idx+=1

        
    def text_tokens(self,text):
        text_seq=[word.text.lower() for word in spacy_eng.tokenizer(text)]
        tokens=[]
        for word in text_seq:
            if word in self.stoi:
                tokens.append(self.stoi[word])
            else:
                tokens.append(self.stoi['<UNK>'])
        return tokens


class FlickrDataset(Dataset):
    def __init__(self,root_dir,image_file,csv_file,caption_col='captions',image_name='images_name',transform=None,delimiter=','):
        self.root_dir=root_dir
        self.image_file=image_file
        self.transform =  transform
        self.dataset=pd.read_csv(os.path.join(self.root_dir,csv_file),delimiter=delimiter)
        
        self.img_names=self.dataset[image_name]
        self.text = self.dataset[caption_col]
        self.vocablary=Vocablary()
        self.vocablary.vocab(self.text)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image_name=self.img_names[index]
        image=Image.open(os.path.join(os.path.join(self.root_dir,self.image_file),image_name))
        if self.transform is not None:
            image=self.transform(image)
        caption=self.text[index]
        token_seq=self.vocablary.text_tokens(caption)
        
        return image,token_seq
    
    
    
class MyCollate():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

transformer=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
dataset= FlickrDataset(root_dir=r'C:\ML\image captioning\flickr30k_images',image_file= 'flickr30k_images',csv_file='results.csv',caption_col= ' comment,,,,,,,,,',image_name= 'image_name',transform=transformer,delimiter= '|')

pad_idx= dataset.vocablary.stoi['<PAD>']

dataloader = DataLoader(dataset,batch_size=32,collate_fn=MyCollate(pad_idx=pad_idx))

for idx, (imgs, captions) in enumerate(dataloader):
    print(imgs.shape)
    print(captions.shape)
