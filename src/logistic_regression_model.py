import torch 
import torch.nn as nn
from src.preprocessing import SUPERCLASSES

class ECGLogisticRegression(nn.Module):
    def __init__(self, num_classes = len(SUPERCLASSES)):
        super().__init__()

        # The ECG has 12 leads and each one measures the heart 1000 times during the test. We use nn.Linear because PyTorch requires us using raw and unscaled numbers.

        self.linear = nn.Linear(in_features=12*1000,out_features = num_classes)

    def forward(self,x):

        # Here we flatten the incoming data, into one single line.
        x = x.reshape(x.size(0), -1)

        # At last we calculate the 5 raw output scores. You will notice that Sigmoid is missing here, but this is because in the "train_logistic_regression.py" file, we use the loss function called "nn.BCEWithLogitsLoss()". 

        # The training file takes the raw scores from this line and pass them through a Sigmoid function. Turning this into a Logistic regression model .

        x = self.linear(x)
        return x
    