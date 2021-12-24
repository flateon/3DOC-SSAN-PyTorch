from datetime import datetime

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import _3DOC_SSAN
from test import test_model

CHANNELS = 103  # 144
NUM_CLASSES = 9  # 15

NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DECAY_STEP = 4
LR_DECAY_RATE = 0.3333


def train_(num_epochs, learning_rate, batch_size, lr_decay_rate, decay_steps):
    # 定义模型
    model = _3DOC_SSAN(CHANNELS, NUM_CLASSES)
    device = "cuda"
    model.to(device)
    cudnn.benchmark = True

    # model.load_state_dict(torch.load('model.pkl'))
    # model = torch.load('models/3DOC_SSAN_0.998327_10-04_17-42.pth')
    # 定义数据集与数据集加载器
    # training_dataset = MyDataset('GRSS2013/train_data.npy', 'GRSS2013/train_labels.npy')
    # testing_dataset = MyDataset('GRSS2013/test_data.npy', 'GRSS2013/test_labels.npy')

    training_dataset = MyDataset('Pavia_University/data_train.npy', 'Pavia_University/labels_train.npy')
    testing_dataset = MyDataset('Pavia_University/data_test.npy', 'Pavia_University/labels_test.npy')

    valid_size = int(0.2 * len(testing_dataset))
    validation_dataset, _ = random_split(testing_dataset, (valid_size, len(testing_dataset) - valid_size))
    training_dataloader = DataLoader(training_dataset, batch_size, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=80, pin_memory=True)
    # 定义损失函数,优化算法及学习率调度器
    criticizer = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=lr_decay_rate)
    writer = SummaryWriter(f'./runs/3DOC_SSAN{datetime.now().strftime("%m-%d_%H-%M")}')
    # 训练模型
    accuracy = [0, ]
    for epoch in range(num_epochs):
        model.train()
        program_bar = tqdm(total=len(training_dataloader), leave=False)
        for i, (images, labels) in enumerate(training_dataloader, start=1):
            images, labels = images.to(device), labels.to(device)
            t_spa, t_spe, t_all = model(images)
            correct = (t_all.argmax(1) == labels).sum().item()
            loss = criticizer(t_all, labels)# + criticizer(t_spa, labels) + criticizer(t_spe, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            program_bar.update()

            if i % 10 == 0:
                iter_times = epoch * len(training_dataloader) + i
                program_bar.set_description_str(f'Epoch:{epoch}')
                program_bar.set_postfix_str(f'loss:{loss.item():.4f}, acc:{correct / len(labels):.4f}')
                writer.add_scalar('training loss', loss.item(), iter_times)
                writer.add_scalar('training acc', correct / len(labels), iter_times)
            # if i % 10 == 0:
            #     tqdm.write(f'Epoch:{epoch}, [{i * batch_size:^8}/{len(training_dataset):^8}] ,loss:{loss.item():.6f}')
        scheduler.step()
        # 验证模型在验证集上的精度
        current_acc = test_model(model, validation_dataloader)
        writer.add_scalar('Validation acc', current_acc, epoch)
        tqdm.write(f'Epoch:{epoch}  Validation_Acc={current_acc * 100:.4f}%')
        # 保存精度提升的模型
        if current_acc > max(accuracy):
            torch.save(model.state_dict(), f"model.pkl")
            tqdm.write("save model")
            if current_acc > 0.998:
                model.eval()
                torch.save(model,
                           f'models/3DOC_SSAN_{current_acc:.6f}_{datetime.now().strftime("%m-%d_%H-%M")}.pth')
        accuracy.append(current_acc)
        writer.close()


if __name__ == '__main__':
    train_(NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, LR_DECAY_RATE, DECAY_STEP)
