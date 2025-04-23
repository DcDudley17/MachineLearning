import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.svm import SVR


class CurveData(Dataset):
    def __init__(self):
        df = pd.read_csv("func2_output.csv")  # Load our dataset

        # Feature and label data
        self.X = torch.tensor(df.iloc[:, 1], dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(df.iloc[:, 2], dtype=torch.float32)

        # Determine the length of the dataset
        self.len = self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X.view(-1)), np.array(self.y)


class CurveFit(nn.Module):
    def __init__(self):
        super(CurveFit, self).__init__()

        self.in_to_h1 = nn.Linear(1, 4)
        self.h1_to_out = nn.Linear(4, 1)

    def forward(self, x):
        x = F.sigmoid(self.in_to_h1(x))
        x = self.h1_to_out(x)
        return x


def trainNN(epochs=100, batch_size=16, lr=0.001, epoch_display=25):
    cd = CurveData()

    # Create a data loader that shuffles each time and allows for the last batch to be smaller
    # than the rest of the epoch if the batch size doesn't divide the training set size
    curve_loader = DataLoader(cd, batch_size=batch_size, drop_last=False, shuffle=True)

    # Create my neural network
    curve_network = CurveFit()

    # Mean square error (RSS)
    mse_loss = nn.MSELoss(reduction='sum')

    # Select the optimizer
    optimizer = torch.optim.Adam(curve_network.parameters(), lr=lr)

    running_loss = 0.0

    for epoch in range(epochs):
        for _, data in enumerate(curve_loader, 0):
            x, y = data

            optimizer.zero_grad()  # resets gradients to zero

            output = curve_network(x)  # evaluate the neural network on x

            loss = mse_loss(output.view(-1), y)  # compare to the actual label value

            loss.backward()  # perform back propagation

            optimizer.step()  # perform gradient descent with an Adam optimizer

            running_loss += loss.item()  # update the total loss

        # every epoch_display epochs give the mean square error since the last update
        # this is averaged over multiple epochs
        if epoch % epoch_display == epoch_display - 1:
            print(
                f"Epoch {epoch + 1} / {epochs} Average loss: {running_loss / (len(cd) * epoch_display):.6f}")
            running_loss = 0.0
    return curve_network, cd


cn, cd = trainNN(epochs=50)

with torch.no_grad():
    y_pred = cn(cd.X).view(-1)
    y_linear = cn(torch.linspace(0,3,steps=1000).view(-1,1)).view(-1)

x_numpy, y_numpy = cd.to_numpy()

print(f"ANN MSE (fully trained): {np.average((y_numpy-np.array(y_pred))**2)}")

svr = SVR()
svr.fit(x_numpy.reshape(-1,1), y_numpy)
print(f"SVR number of support vectors: {svr.support_vectors_.shape}")
print(f"SVR MSE: {np.average((svr.predict(x_numpy.reshape(-1,1))-y_numpy)**2.0)}")

x = np.linspace(0,3,num=1000)
plt.scatter(x_numpy,y_numpy,s=1.0)

#for our function
plt.plot(x,np.sin(x**x)/2**((x**x-np.pi/2)/np.pi),c='black')

svr_out = svr.predict(x.reshape(-1,1)).reshape(-1)
plt.plot(x, svr_out, c = 'g')
plt.plot(x, y_linear.numpy(), c='red', label='Neural Net')
plt.legend()
plt.show()

