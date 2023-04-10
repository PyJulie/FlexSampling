import torch
from collections import Counter
import numpy as np
import torch.nn as nn

beta = 0.9999

train_image_datasets = '' #define train_image_datasets here
c = Counter(train_image_datasets.img_label)
c = c.most_common()
cls_num_list = []

for i in c:
    cls_num_list.append(i[1])
    
effective_num = 1.0 - np.power(beta, cls_num_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
train_criterion = nn.CrossEntropyLoss(weight=per_cls_weights)