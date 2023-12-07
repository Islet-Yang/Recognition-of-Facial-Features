# Recognition-of-Facial-Features
This is one of the course assignments of 2023 Autumn Introduction to Artificial Intelligence of the Department of Automation, Tsinghua University.The background is given 50,000 face images with size in 178_218_3 , corresponding to 40 face features and labels. We need to do an image classification task on it.
    
Specific task implementation and modules explanations are as follows:  
    
## MyDataset.py
Class MyDataset inherits torch.Dataset, its main function is to store images and labels properly  
Sepcific functions:    
  * a. Load： Save the data from the file in the dataset and extract the characteristics we need.
  * b. Process：  We stipulate that 1 means the feature is present and -1 means it is not present
  * c. Index：  Using `__getitem__()` to call by index.

## Earlystopping.py
Class EarlyStopping is a tool to save the best model in the training process. If there is no performance improvement on several epochs at validation-set, the program can be terminated early.  
  
## ResNet.py
I chose ResNet50 as the training network. It is verified that only one epoch network is needed to achieve 91.38% accuracy for binary classification task and 82.79% accuracy for five-classification task. 

## DenseConv.py
I designed a multi-layer CNN network as an auxiliary test without careful tuning. Setting batch_size to 1 can be used to classify in general effects. 
  
## Main.py
This is the main program. It contains training and testing process. Every parameter and action is clearly annotated, and you can check it in the code.  

## Supplementary instruction
The runtime program needs to specify the classification characteristics with arg. Here are two examples:
python Main.py --characteristic_list Smiling Others
python Main.py --characteristic_list Black_Hair Blond_Hair Brown_Hair Gray_Hair Others
