import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, root_dir, csv_file, characteristic_list, num_classes, transform=None):
        '''
        Initialize the dataset
        '''
        self.root_dir = root_dir
        self.transform = transform # image formatting
        self.images = os.listdir(root_dir)
        self.characteristic_list = characteristic_list
        self.num_classes = num_classes
        self.chacteristic_rank = self.call_by_rank(csv_file)               
        self.labels = self.load_csv(csv_file)

    def call_by_rank(self, csv_file):
        '''
        Transform the characteristic list to the corresponding rank in the csv file
        '''
        df = pd.read_csv(csv_file)
        row_0 = df.columns[1:].tolist()
        characteristic_rank = []
        for i, characteristic in enumerate(self.characteristic_list[:self.num_classes-1]):
            if characteristic not in row_0:
                # check if all the characteristics are in the csv file
                raise ValueError(f'{characteristic} not in {csv_file}')
            else:
                characteristic_rank.append(row_0.index(characteristic))

        return characteristic_rank
      
    def load_csv(self, csv_file):
        '''  
        Load the csv file and return a dictionary with filename as key and label as value
        '''
        df = pd.read_csv(csv_file)
        labels = {}
        for i, row in df.iterrows():
            filename = row[0]
            label_vector = row[1:].values.tolist()
            new_label = self.process_label(label_vector)
            labels[filename] = new_label
            
        return labels

    def process_label(self, label_vector):
        '''
        Distinguish the label of the image
        '''
        for rank, i in enumerate(self.chacteristic_rank):
            if label_vector[i] == 1:
                return rank
        return self.num_classes - 1

    def __len__(self):
        '''
        Return the length of the dataset
        '''
        return len(self.images)

    def __getitem__(self, idx):
        '''
        Return the image and its label
        Call by idx
        '''
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        labels = self.labels[self.images[idx]]
        return image, torch.tensor(labels, dtype=torch.long)

# Test the dataset           
if __name__ == '__main__':
    transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    dataset=MyDataset(root_dir='./data_face_imgs/images',
                    csv_file='./data_face_imgs/anno.csv',
                    #characteristic_list=['Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Others'],
                    characteristic_list = ['Smiling','Others'],
                    num_classes=2,
                    transform=transforms)
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])
