import torch
from torch.autograd import Variable

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_loss(inputs, labels, model_ft, optimizer_ft):
    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
    optimizer_ft.zero_grad()

    outputs = model_ft(inputs)
    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
    return loss