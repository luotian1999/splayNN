import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from time import gmtime, strftime
import argparse
import sys

class Net(nn.Module):
    def __init__(self, batch_size, LSTM1_hidden_size, LSTM2_hidden_size, encoder_hidden_size, encoder_hidden_size2, count_hidden_size):
        super(Net, self).__init__()
        self.IP_LSTM = nn.LSTM(1, LSTM1_hidden_size, batch_first=True)
        self.IP_encode1 = nn.Linear(4*LSTM1_hidden_size, encoder_hidden_size)
        self.IP_encode2 = nn.Linear(encoder_hidden_size, encoder_hidden_size2)
        self.Seq_LSTM = nn.LSTM(encoder_hidden_size2, LSTM2_hidden_size, batch_first=True)
        self.count1 = nn.Linear(LSTM2_hidden_size+4, count_hidden_size)
        self.count2 = nn.Linear(count_hidden_size, 1)
        self.LSTM1_hidden_size = LSTM1_hidden_size
        self.LSTM2_hidden_size = LSTM2_hidden_size
        self.batch_size = batch_size

        #self.simple1 = nn.Linear(20, count_hidden_size)
        #self.simple2 = nn.Linear(count_hidden_size, 1)

    def forward(self, x, ele, device):

        
        LSTM2_hidden = (torch.randn(1,self.batch_size,self.LSTM2_hidden_size).to(device),torch.randn(1,self.batch_size,self.LSTM2_hidden_size).to(device))
        for i in range(len(x[0])):
            LSTM1_hidden = (torch.randn(1,self.batch_size,self.LSTM1_hidden_size).to(device), torch.randn(1,self.batch_size,self.LSTM1_hidden_size).to(device))
            LSTM1_output, LSTM1_hidden = self.IP_LSTM(x[:,i].view(self.batch_size, -1, 1), LSTM1_hidden)
            encoding = self.IP_encode1(LSTM1_output.reshape(self.batch_size, 1, -1))
            encoding = F.relu(encoding)
            encoding = self.IP_encode2(encoding)
            LSTM2_output, LSTM2_hidden = (encoding, LSTM2_hidden)
        output = LSTM2_output.view(-1,1,1)
        output = torch.cat((output, ele.view(-1,1,1))).view(1,1,-1)
        output = self.count1(output)

        output = F.relu(output)
        output = self.count2(output)
        
        
        #output = self.simple1(torch.cat((x.view(-1,1,1), ele.view(-1,1,1))).flatten().view(1,1,-1))
        #output = F.relu(output)
        #output = self.simple2(output)

        return output.view(-1,1)

class ProcessedDataset(Dataset):
    def __init__(self, data, seqlen):
        self.data = data
        self.seqlen = seqlen

    def __getitem__(self, index):
        x = self.data['x'][index:index+self.seqlen-1]
        x = x.astype('float32')
        x = torch.from_numpy(x)
        y = self.data['y'][index+self.seqlen-1:index+self.seqlen]
        y = y.astype('float32')
        #y = np.log1p(y)
        y = torch.from_numpy(y)
        z = self.data['x'][index+self.seqlen-1:index+self.seqlen]
        z = z.astype('float32')
        z = torch.from_numpy(z)
        return x,y,z

    def __len__(self):
        return len(self.data)-self.seqlen
        #return 1000


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train", type=str, help="npy file", default="")
    argparser.add_argument("--test", type=str, help="npy file", default="")
    argparser.add_argument("--seqlen", type=int, help="How many previous queries to look at", default=5)
    argparser.add_argument("--LSTM1_hidden_size", type=int, help="", default=16)
    argparser.add_argument("--LSTM2_hidden_size", type=int, help="", default=16)
    argparser.add_argument("--encoder_hidden_size", type=int, help="", default=16)
    argparser.add_argument("--encoder_hidden_size2", type=int, help="", default=16)
    argparser.add_argument("--count_hidden_size", type=int, help="", default=16)
    argparser.add_argument("--lr", type=float, help="learning rate", default=0.01)
    argparser.add_argument("--epochs", type=int, help="number of epochs", default=5)
    argparser.add_argument("--batch_size", type=int, help="number of epochs", default=1)
    args = argparser.parse_args()

    train_data = np.load(args.train)
    test_data = np.load(args.test)

    for i in range(20):
        print(train_data['y'][i])

    model = Net(args.batch_size, args.LSTM1_hidden_size, args.LSTM2_hidden_size, args.encoder_hidden_size, args.encoder_hidden_size2, args.count_hidden_size)
    print(model)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #if torch.cuda.device_count() > 1:
    #    print("GPUs %d" % torch.cuda.device_count())
    #    model = nn.DataParallel(model)
    model.to(device)


    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr)

    train_loader = DataLoader(dataset=ProcessedDataset(train_data, args.seqlen), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=ProcessedDataset(test_data, args.seqlen), batch_size=args.batch_size, shuffle=True)

    best_loss = float("inf")


    for e in range(args.epochs):
        print("Epoch: %d" % (e+1))
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            inputs, targets, ele = data

            inputs = inputs.to(device)
            ele = ele.to(device)
            targets = targets.to(device)

            outputs = model(inputs, ele, device)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #if i % 2000 < 10:


            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('TRAINING:[Epoch %d, %5d] loss: %.3f' %
                    (e + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                print(outputs)
                print(targets)

            #break
        
        testing_loss = 0.0
        running_loss = 0.0
        #break
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(train_loader, 0):
                inputs, targets, ele = data

                inputs = inputs.to(device)
                ele = ele.to(device)
                targets = targets.to(device)

                outputs = model(inputs, ele, device)
                loss = loss_fn(outputs, targets)

                running_loss += loss.item()
                testing_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('TESTING:[Epoch %d, %5d] running loss: %.3f, testing_loss: %.3f' %
                        (e + 1, i + 1, running_loss / 2000, testing_loss))
                    running_loss = 0.0
            if testing_loss < best_loss:
                best_loss = testing_loss
                path = "/home/tian/splaynn/" + strftime("%m-%d-%H:%M", gmtime())+"+"+str(args.seqlen) + ".pth"
                print("Best model so far, saving to " + path)
                torch.save(model.state_dict(), path)





