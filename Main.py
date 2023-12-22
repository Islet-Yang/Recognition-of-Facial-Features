import numpy as np
import random
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from MyDataset import MyDataset
from ResNet import ResNet
from DenseConv import DenseConv
from EarlyStopping import EarlyStopping
from Conv import Conv
from sklearn.metrics import precision_score, recall_score, f1_score

class ModelWorking:
    def __init__(self, characteristic_list):
        '''
        Initialize all parameters, data sets, and model
        '''
        self.seed = 19  # seed number
        self.set_seed(self.seed)  # Fix the seed
        self.image_path = './data_face_imgs/images'  # image-path
        self.label_path = './data_face_imgs/anno.csv'  # file-path of the labels        
        
        self.batch_size = 32  # batch_size
        self.epoch = 500  # epochs to train
        self.learning_rate = 3e-4 # learning rate
        self.weight_decay = 1e-5 # weight decay
        self.warm_up = 0.01 # warm up
        self.characteristic_list = characteristic_list # list of characteristics
        self.num_classes = len(self.characteristic_list)  # number of classes
        self.train_patience = 30  # patience for early stopping
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check GPU and use it
        
        # Here choose the model you want to use
        # self.model = ResNet(self.num_classes).to(self.device) # Initialize the model
        self.model = Conv(self.num_classes).to(self.device) #Initialize the model
        self.criterion = nn.CrossEntropyLoss()  # Initialize the loss function
        
        # Using Adam as optimizer as an example. SGD is another common choice.
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay) # Initialize the optimizer
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch // 10)) # For warm_up
        
        self.save_path = 'best_model.pth'  # save-path for the model
        self.log_file_path = 'log_'+self.model.getname()+'_'+str(self.num_classes)+'.txt'  # save-path for the log file
        
        self.early_stopping = EarlyStopping(self.train_patience, checkpoint_path=self.save_path, mode='max') # Initialize the early_stopping object
        
        self.transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Resize the image to 224x224 (required by ResNet)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image
        ])      

        # divide train/validation/test dataset
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1

        self.dataset = MyDataset(self.image_path,self.label_path,self.characteristic_list,
                                 self.num_classes,transform=self.transforms) # Create dataset
        
        self.total_len = len(self.dataset)
        self.train_len = int(self.total_len * self.train_ratio)
        self.valid_len = int(self.total_len * self.val_ratio)
        self.test_len = int(self.total_len * self.test_ratio)
        self.train_set, self.valid_set, self.test_set = random_split(self.dataset, [self.train_len, self.valid_len, self.test_len])
        
        # Create data_loader
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)
        
    def evaluate_model(self, loader, model, criterion, device):
        '''
        Evaluate the model on the given data loader
        Loss: Using CrossEntropyLoss
        Accuracy: Count the number of correct predictions and divide by the number of total predictions
        '''
        model.eval()
        total_loss = 0

        with torch.no_grad():
            correct = 0
            total = 0
            predictions = []
            ground_truth = []
            progress_bar = tqdm(enumerate(loader), total=len(loader), desc='Evaluation')

            for batch_idx, (images, labels) in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels) # Calculate the loss
                total_loss += loss.item() * images.size(0)

                # Update progress bar description
                progress_bar.set_postfix({'Loss': total_loss / total, 'Accuracy': 100 * correct / total})
                
                # Collect predictions and ground truth for precision, recall, f1-score
                predictions.extend(predicted.cpu().numpy())
                ground_truth.extend(labels.cpu().numpy())

        # Calculate various metrics
        accuracy = 100 * correct / total # Calculate the accuracy
        precision = precision_score(ground_truth, predictions, average='weighted') # Calculate the precision
        recall = recall_score(ground_truth, predictions, average='weighted') # Calculate the recall
        f1 = f1_score(ground_truth, predictions, average='weighted') # Calculate the f1-score
        
        return total_loss / len(loader.dataset), accuracy, precision, recall, f1
  
    def train(self):
        '''
        Train the model
        Early stopping is used to stop training when the validation accuracy stops increasing
        '''
        for epoch in range(self.epoch):
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Training') 
            self.scheduler.step()          
            for batch_idx, (images, labels) in progress_bar:
                self.model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
            print(f'Epoch [{epoch+1}/{self.epoch}], Loss: {loss.item()}')
            validation_loss, validation_accuracy, validation_precision, validation_recall, validation_f1 = self.evaluate_model(self.validation_loader, self.model, self.criterion, self.device) # Evaluate the model on the validation set
            
            # Save the log file
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(f'Epoch [{epoch+1}/{self.epoch}]\n')
                log_file.write(f'Total Loss: {validation_loss}, Accuracy: {validation_accuracy}, Precision:{validation_precision}, Recall:{validation_recall}, F1-score:{validation_f1}\n')
                
            print('Validation Loss:{}'.format(validation_loss))
            print('Validation Accuracy: {}'.format(validation_accuracy))

            # Early stopping
            self.early_stopping.step(validation_accuracy, self.model)
            if(self.early_stopping.should_stop()):
                print('Early Stopping is Triggered.')
                self.epoch = epoch + 1
                break
            else:
                print('Early Stopping count: %s / %s'%(self.early_stopping.now_count(),self.train_patience))
        
    def test(self):
        '''
        Test the model on the test set
        '''
        self.early_stopping.load_checkpoint(self.model) # Load the best model
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = self.evaluate_model(self.test_loader, self.model, self.criterion, self.device) # Evaluate the model on the test set
        print('Test Loss:{}'.format(test_loss))
        print('Test Accuracy: {}'.format(test_accuracy))
        
        # Save the log file
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f'\nFinal Test:\n')
            log_file.write(f'Total Loss: {test_loss}, Accuracy: {test_accuracy}, Precision:{test_precision}, Recall:{test_recall}, F1-score:{test_f1}\n')
        
    def set_seed(self, seed):
        '''
        Fix the seed for all random factors
        '''
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

# Main program
if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Classification Characteristics')
    parser.add_argument('--characteristic_list', nargs='+', default=['Smiling', 'Others'], help='Input list')
    args = parser.parse_args()
    
    # Start training and testing
    analysis = ModelWorking(args.characteristic_list)
    analysis.train()
    analysis.test()
