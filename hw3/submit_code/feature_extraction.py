class VGG_fe(nn.Module):

    def __init__(self, num_layers, num_features):
        super().__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        self.num_features = num_features

        for param in vgg16.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(vgg16.features.children())[:num_layers])
        self.fc = nn.Sequential(nn.Linear(in_features=num_features, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=256))


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_features)
        x = self.fc(x)
        return x

def train_model(model, dataset, criterion, optimizer, num_epochs, batch_size, scheduler=None):
    start_time = time.time()
    model.train(True)
    dataset_size = dataset.__len__()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    record ={'loss':[], 'acc':[]}

    for epoch in range(num_epochs):
        if scheduler is not None:
            scheduler.step()
        running_loss = 0.
        running_corrects = 0

        for data in dataloader:
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size
        record['loss'].append(epoch_loss)
        record['acc'].append(epoch_acc)
        print('%d epoch loss: %f    accuracy: %f%%' % (epoch, epoch_loss, epoch_acc*100))

    model.train(False)
    time_elapsed = time.time() - start_time
    print('Training comple in %dm, %ds' % (time_elapsed//60, time_elapsed%60))
    return model, record

def test_model(model, dataset):
    testloader = DataLoader(dataset, batch_size=16)
    correct_cnt = 0

    for data in testloader:
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct_cnt += torch.sum(preds.data == labels.data)

    acc = correct_cnt / dataset.__len__()
    print('Test Set Accuracy: %f%%' % (acc*100))
    return acc

def each_class_accuracy(model, dataset, num_classes=256):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    testloader = DataLoader(dataset, batch_size=16)

    for data in testloader:
        images, labels = data
        outputs = model(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.cuda())
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(num_classes):
        print('Accuracy of class %d : %2d %%' % (i, 100*class_correct[i]/class_total[i]))

def plot_cost_acc(record):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    tc_plt, = ax1.plot(record['acc'], label='Training Accuracy')
    vc_plt, = ax1.plot(record['loss'], label='Training Loss')
    ax1.legend(handles=[tc_plt, vc_plt])
    plt.show()

def plot_acc(acc):
    acc = np.array(acc) * 100
    print(acc)
    plt.plot(acc)
    plt.ylabel('Accuracy %')
    plt.xlabel('Epochs')
    plt.show()

def plot_loss(loss):
    plt.plot(loss)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()
