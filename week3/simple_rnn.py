import torch
import random
import torch.nn as nn


'''
n分类
abcde 0
bacde 1
bcade 2
bcdae 3
bcdea 4
'''
class MyRnnModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_classes):
        super(MyRnnModel,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.rnn = nn.RNN(embedding_dim,hidden_dim,batch_first=True)
        self.layer = nn.Linear(hidden_dim,num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x,label=None):
        x = self.embedding(x) # [batch_size,seq_len,embedding_dim]
        x,_ = self.rnn(x) # [batch_size,seq_len,hidden_dim]
        x = x[:,-1,:] # [batch_size,hidden_dim]
        y_pred = self.layer(x)

        if label is not None:
            return self.loss(y_pred,label)
        else:
            return y_pred

def main():
    vocab = build_vocab()
    epouch = 100
    embedding_dim = 256
    hidden_dim = 256
    num_classes = 5
    model = MyRnnModel(len(vocab),embedding_dim,hidden_dim,num_classes)
    optim = torch.optim.Adam(model.parameters(),lr=0.001)
    for i in range(epouch):
        model.train()
        x,y = build_samples(100,vocab)
        loss = model(x,y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(i,loss.item())
    return model

def build_vocab():
    vocab = {}
    alphas = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    for index,alpha in enumerate(alphas):
        vocab[alpha] = index
    return vocab
def build_samples(nums,vocab):
    alphas = [chr(i) for i in range(ord('a'), ord('z') + 1)]

    samples = []
    labels = []
    for i in range(nums):
        sample = random.sample(alphas, 5)
        if 'a' not in sample:
            sample[random.randint(0, 4)] = 'a'
        #在sample中找到'a'的位置
        label = sample.index('a')
        samples.append([vocab[alpha] for alpha in sample])
        labels.append(label)
    return torch.LongTensor(samples), torch.LongTensor(labels)



if __name__ == '__main__':
    model = main()

    model.eval()
    print(torch.argmax(model(torch.LongTensor([[0,1,2,3,4]]))))
    print(torch.argmax(model(torch.LongTensor([[1,1,2,0,4]]))))
    print(torch.argmax(model(torch.LongTensor([[5,1,0,3,4]]))))
    print(torch.argmax(model(torch.LongTensor([[8,1,9,0,4]]))))