import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

def loader(path):
    return Image.open(path)

class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, padding='same'
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 35 * 35, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)
        return outputs
# Training function
def train(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []


    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        # compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

        if idx % log_interval == 0 and idx > 0:

            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss


# Evaluation function
def evaluate(model, criterion, valid_dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss

if __name__ == '__main__':
    data_paths = {
    'train': './train',
    'valid': './validation',
    'test': './test'
}
    
    img_size = 150
    train_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(
        root=data_paths['train'],
        loader=loader,
        transform=train_transforms
    )

    valid_data = datasets.ImageFolder(
        root=data_paths['valid'],
        transform=train_transforms
    )

    test_data = datasets.ImageFolder(
        root=data_paths['test'],
        transform=train_transforms
    )

    BATCH_SIZE = 512

    train_dataloader = data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=BATCH_SIZE
    )
    valid_dataloader = data.DataLoader(
        valid_data,
        batch_size=BATCH_SIZE
    )
    test_dataloader = data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE
    )

# load image from path
    num_classes = len(train_data.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lenet_model = LeNetClassifier(num_classes)
    lenet_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 2e-4
    optimizer = optim.Adam(lenet_model.parameters(), learning_rate)

    num_epochs = 10
    save_model = './model'

    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100

    for epoch in range(1, num_epochs+1):

        # Training
        train_acc, train_loss = train(lenet_model, optimizer, criterion, train_dataloader, device, log_interval=10)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Evaluation
        eval_acc, eval_loss = evaluate(lenet_model, criterion, valid_dataloader)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Save best model
        if eval_loss < best_loss_eval:
            torch.save(lenet_model.state_dict(), save_model + '/lenet_model.pt')

        # Print loss, acc end epoch
        print("=" * 59)
        print(
            "| End of epoch {:3d} |  Train Accuracy {:8.3f} | Train Loss {:8.3f} "
            "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f}".format(
                epoch, train_acc, train_loss, eval_acc, eval_loss
            )
        )
        print("=" * 59)

    # Load best model
    lenet_model.load_state_dict(torch.load(save_model + '/lenet_model.pt'))
    lenet_model.eval()
