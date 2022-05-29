import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from multi_output_FCN import Net
from load_mel import MelDataset, ToTensor

start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



batch_size = 2 # 每次抽取四个sample运算 == mini batch



dataset = MelDataset(annotation="data/genre_ins_key.csv",audio_info="data/clip_info_with mel_data_path.csv",
                                    transform=ToTensor())
#indices = np.random.permutation(len(dataset))[:100]
#dataset = torch.utils.data.Subset(dataset, indices)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset,
                                          shuffle=True, num_workers=4,batch_size=batch_size,drop_last=True)#drop_last
#https://stackoverflow.com/questions/56576716/pytorch-dataloader-fails-when-the-number-of-examples-are-not-exactly-divided-by
testloader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=True, num_workers=4,batch_size=batch_size,drop_last=True)


net = Net()
net.to(device)  # send network to GPU

# define loss function
criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9)  # Implements stochastic gradient descent (optionally with momentum).

# train:
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['mel'].to(device,dtype=torch.float), data['label']  # send to GPU

        #print("input: ",inputs.size(),labels.size())

        #label_genre = labels[0:39].flatten().to(device,dtype=torch.float)
        label_genre = labels[torch.arange(labels.size(0)), 0:39].to(device,dtype=torch.float)
        label_ins = labels[torch.arange(labels.size(0)), 39:117].to(device,dtype=torch.float)
        label_key = labels[torch.arange(labels.size(0)), 117].to(device,dtype=torch.float)
        #print(label_genre.size(),label_ins.size(),label_key.size())


        # zero the parameter gradients, otherwise, the gradient would be a combination of the old gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        genre, ins, key = net(inputs)  # forward
        #print("predicted: ",genre,ins,key)

        loss1 = criterion(genre, label_genre) #mse
        loss2 = criterion(ins, label_ins) #mse
        loss3 = criterion(key,label_key) #independent
        #print("return",genre.size(),ins.size(),key.size())
        loss = loss1 + loss2 + loss3

        #print("predicted ins: ", ins)
        # print("loss: ",loss1," ",loss2," ",loss)
        # This criterion computes the cross entropy loss between input and target. 错误率
        loss.backward()  # backward propagation
        optimizer.step()  # gradient decent to minimize cost function. All optimizers implement a step() method, that updates the parameters.

        # print statistics
        running_loss += loss.item()  # extracts the loss’s value as a Python float.
        if i % 200 == 199:  # print every 200 mini-batches/batch size, every 200 samples' loss in the whole dataset
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

PATH = './trained_genre_ins_key.pth'
torch.save(net.state_dict(), PATH)
print('Finished Training')

# test


net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        inputs, labels = data['mel'].to(device, dtype=torch.float), data['label']  # send to GPU

        #print(labels)

        label_genre = labels[torch.arange(labels.size(0)), 0:39].to(device, dtype=torch.float)
        label_ins = labels[torch.arange(labels.size(0)), 39:117].to(device, dtype=torch.float)
        label_key = labels[torch.arange(labels.size(0)), 117].to(device, dtype=torch.float)
        print(label_key.size())
        genre, ins, key = net(inputs)  # forward

        predicted_genre = genre
        predicted_ins = ins
        predicted_key = key
        print(predicted_genre.size())
        print("predicted: ",predicted_genre)
        total += len(labels.flatten()) # how many labels are in total (sum every batched label group*batch)
        #dim=1 to not sum batch
        correct += (predicted_genre == label_genre).sum(dim=1) + (
                predicted_ins == label_ins).sum(dim=1)+ (
                           predicted_key == label_key)
        # sum correct predictions in every batch, .item() turn it into a normal number
summ=correct.sum().item()
print("correct: ",correct," sum: ",total)
print('Accuracy of the network on the ', str(test_size), f' test MELs: {100 * summ // total} %')

print("--- %s seconds ---" % (time.time() - start_time))
