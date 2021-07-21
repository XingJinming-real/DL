import torch
import torchvision
from torchvision import datasets
from torch import nn
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale

inSize = 256
outSize = 1
l1 = int(inSize * 0.8)
l2 = int(inSize * 0.6)
l3 = int(inSize * 0.2)
lr = 0.0002
p1 = 0
p2 = 0
p3 = 0
batchSize = 256
weightDecay = 0.9
epochNum = 30
priceScaler = 1000000
TaxValueScaler = 1000000
AnnualScaler = 10000
ListedPriceScaler = 1000000


def getData():
    #     返回训练集和测试姐Iter
    global inSize
    train, test = pd.read_csv("E:\\kaggle\\california-house-prices\\train.csv"), \
                  pd.read_csv("E:\\kaggle\\california-house-prices\\test.csv")
    label = torch.tensor(train['Sold Price'].values / priceScaler, dtype=torch.float32)
    train.drop('Sold Price', axis=1, inplace=True)
    Data = pd.concat([train, test])
    Data.drop(['Id', 'Address', 'Summary', 'Elementary School', 'Middle School', 'High School',
               'Listed On', 'Appliances included', 'Parking', 'Parking features', 'Last Sold On', 'Zip', 'State'],
              axis=1,
              inplace=True)
    le = LabelEncoder()
    for i in ['Heating', 'Cooling', 'Bedrooms', 'Flooring'
        , 'Heating features', 'Cooling features',
              'Laundry features', ]:
        Data[i] = le.fit_transform(Data[i].astype(str))
    # Data['Tax assessed value'] = Data["Tax assessed value"] / TaxValueScaler
    # Data['Annual tax amount'] = Data["Annual tax amount"] / AnnualScaler
    # Data['Listed Price'] = Data["Listed Price"] / ListedPriceScaler
    # Data['Last Sold Price'] = Data['Last Sold Price'] / ListedPriceScaler
    sipInput = SimpleImputer(strategy="most_frequent")
    Data.columns = range(Data.shape[1])
    for i in range(Data.shape[1]):
        if Data[i].isna().any():
            if Data[i].dtype != object:
                Data[i] = scale(Data[i])
                Data[i] = Data[i].fillna(0)
        elif Data[i].dtype == 'int32' or Data[i].dtype == 'float':
            Data[i] = scale(Data[i])
    Data = pd.get_dummies(Data).values
    inSize = Data.shape[1]
    # pd.DataFrame(Data).to_csv('./CA-house-prices.csv')
    trainData = torch.tensor(Data[:len(label), :], dtype=torch.float32)
    testData = torch.tensor(Data[len(label):, :], dtype=torch.float32)
    trainData = data.TensorDataset(*(trainData, label))
    return data.DataLoader(trainData, batch_size=batchSize, shuffle=True), testData


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(inSize, l1),
                                   nn.ReLU(),
                                   nn.Dropout(p1),
                                   nn.Linear(l1, l2),
                                   nn.ReLU(),
                                   nn.Dropout(p2),
                                   nn.Linear(l2, l3),
                                   nn.ReLU(),
                                   nn.Dropout(p3),
                                   nn.Linear(l3, outSize))

    def forward(self, X):
        return self.model(X)


def init_params(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()


def main():
    trainIter, testData = getData()
    net = model()
    # net.to('cuda')
    net.apply(init_params)
    loss = nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr, weight_decay=weightDecay)
    for epoch in range(epochNum):
        print(f'epochNum: {epoch}')
        L = 0
        for X, y in trainIter:
            # X = X.cuda()
            # y = y.cuda()
            opt.zero_grad()
            y_hat = net(X).flatten()
            l = loss(y_hat, y)
            l.backward()
            L += float(l)
            opt.step()
        print("loss: ", L)
    # testData = testData.cuda()
    pre = net(testData) * priceScaler
    pre = pd.DataFrame(pre.flatten().detach().data)
    pre.to_csv('CAHousePricePrediction')
    torch.save(net, 'CAPricePrediction')


if __name__ == '__main__':
    main()
