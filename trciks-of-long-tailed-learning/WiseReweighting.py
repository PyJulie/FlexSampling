import torch
from collections import Counter
import numpy as np
import torch.nn as nn

train_image_datasets = '' #define train_image_datasets here
beta = 0.9999
c = Counter(image_datasets['train'].img_label)
c = c.most_common()
cls_num_list = []
for i in c:
    cls_num_list.append(i[1])
effective_num = 1.0 - np.power(beta, cls_num_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights_ = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)

# Calculate the validation accuracy first.
neg_acc_per_class = [1-(val_correct[n]/val_correct[n]) for n in range(num_class)]
new_weights = np.array(new_weights)*per_cls_weights_
new_weights = torch.FloatTensor(new_weights).cuda()
train_criterion = nn.CrossEntropyLoss(weight=new_weights)
