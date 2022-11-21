## Store activation-embedding representation of test images

import pandas as pd
import torch
from torchvision import datasets, transforms
from main import Net

# load model
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

# init data frame, where we save the intermediate representation
activations = pd.DataFrame()
outputs = pd.DataFrame()
labels = pd.DataFrame()

# Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST('../data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64)
device = torch.device("cpu")

# perform forward pass, save representation in the data frame
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        log_softmax, intermediate = model(data)
        # add both to DataFrame
        activations = activations.append(pd.DataFrame(intermediate))
        outputs = outputs.append(pd.DataFrame(log_softmax))
        labels = labels.append(pd.DataFrame(target))


activations.to_csv('test_activations.csv')
outputs.to_csv('test_log_softmax.csv')
labels.to_csv('test_labels.csv')
