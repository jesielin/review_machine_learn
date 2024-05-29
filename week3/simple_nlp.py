import  torch
import  torch.nn as nn
import random

'''
a
abcd 正样本
cdef 负样本
'''

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(6, 100)
        self.pool = nn.MaxPool1d(4)
        self.fc = nn.Linear(100, 2)
        self.loss = nn.CrossEntropyLoss()


    def forward(self,x,label=None):
        x = self.embedding(x)
        x = x.transpose(1,2)
        x = self.pool(x)
        x = x.squeeze(2)
        y_pred = self.fc(x)

        if label is not None:
            loss = self.loss(y_pred,label)
            return loss
        else:
            return torch.softmax(y_pred,dim=-1)

vocab = {
    'a':0,
    'b':1,
    'c':2,
    'd':3,
    'e':4,
    'f':5
}
def build_sample(nums):
    alphas = ['a', 'b', 'c', 'd', 'e', 'f']

    inputs = []
    labels = []

    for i in range(nums):
        raw_alpha = random.sample(alphas, 4)
        x = [vocab[alpha] for alpha in raw_alpha]
        label =  0 if 'a' in raw_alpha else 1
        inputs.append(x)
        labels.append(label)

    return torch.LongTensor(inputs),torch.LongTensor(labels)


def main():
    model = MyModel()
    epoch = 500
    optim = torch.optim.Adam(model.parameters(),lr=0.001)

    for i in range(epoch):
        inputs,labels = build_sample(10)
        loss = model(inputs,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss.item())

    return model


if __name__ == '__main__':
    model = main()
    print(torch.argmax(model(torch.LongTensor([[0,1,2,3]]))))
    print(torch.argmax(model(torch.LongTensor([[1,1,2,3]]))))
    print(torch.argmax(model(torch.LongTensor([[4,1,2,3]]))))
    print(torch.argmax(model(torch.LongTensor([[0,1,2,3]]))))
    print(torch.argmax(model(torch.LongTensor([[0,1,2,3]]))))