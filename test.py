import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.backends.cudnn import benchmark
from dataset import MyDataset
import numpy as np
from sklearn import metrics


def test_model(model, dataloader):
    with torch.no_grad():
        # 判断模型在GPU还是CPU
        model.eval()
        device = next(model.parameters()).device
        correct, total = 0, 0
        for images, labels in tqdm(dataloader, leave=False):
            images, labels = images.to(device), labels.to(device)
            _, _, t_all = model(images)
            correct += (t_all.argmax(1) == labels).sum()
            total += len(labels)
            # print(f'The Accuracy of {total} images is {correct / total * 100:.6f}%')
        model.train()
    return correct.item() / total


def test_oa_aa_kappa(net, test_loader):
    net.eval()
    all_predict = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.to('cuda')
            outputs = net(inputs)
            predict = outputs[-1].argmax(1)

            all_predict.append(predict.cpu().numpy())
            all_targets.append(targets.numpy())

        all_predict = np.concatenate(all_predict)
        all_targets = np.concatenate(all_targets)

        oa = metrics.accuracy_score(all_targets, all_predict)
        confusion_mat = metrics.confusion_matrix(all_targets, all_predict)
        aa = (np.diagonal(confusion_mat) / confusion_mat.sum(1)).sum() / len(confusion_mat)
        kappa = metrics.cohen_kappa_score(all_predict, all_targets)
        return oa, aa, kappa, confusion_mat


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark = True
    testing_dataloader = DataLoader(MyDataset('Pavia_University/data_test.npy', 'Pavia_University/labels_test.npy'),
                                    128, num_workers=0, pin_memory=True)
    result = []
    # 加载模型参数
    # for file in Path('models').glob('*.pth'):
    #     model = torch.load(file).to(device)
    #     result.append((file.name, test_model(model, testing_dataloader)))
    #     print(result[-1], '\n')
    #     del model
    # result.sort(key=lambda x: x[1], reverse=True)
    # for model in result:
    #     print(f'{model[0]}      Test:{model[1] * 100:.6f}%')
    model = torch.load('models/3DOC_SSAN_0.998455_12-05_11-01.pth').to(device)
    result = test_oa_aa_kappa(model, testing_dataloader)
