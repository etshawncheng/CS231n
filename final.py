from collections import deque
import torch
from torch import nn, optim, Generator
from torch.utils.data import DataLoader, Dataset, random_split
from numpy.random import choice
from typing import Iterable, Callable, Type, Optional, Union, Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from operator import mul


def product(nums: Iterable[Type], func: Callable[[Type, Type], Type] = mul) -> Type:
    """return product of iterable"""
    _it = iter(nums)
    v: Type = next(_it)
    for _v in _it:
        v = func(v, _v)
    return v


class DS_SalaryDataset(Dataset):
    """DS Salary dataset."""

    def __init__(self):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ds_salaries: pd.DataFrame = pd.read_csv("./data/ds_salaries.csv")
        self.nonnumerical_column_encoders = {
            c: LabelEncoder() for c, dt in ds_salaries.dtypes.items() if dt == 'O'}
        ds_salaries[list(self.nonnumerical_column_encoders.keys())] = pd.DataFrame(
            e.fit_transform(ds_salaries[c]) for c, e in self.nonnumerical_column_encoders.items()).T
        #
        self.feature = torch.from_numpy(
            ds_salaries[ds_salaries.columns[:4].append(ds_salaries.columns[7:])].to_numpy(dtype=np.float32))
        # two status
        self.salary = torch.reshape(torch.tensor(
            ds_salaries.salary_in_usd.to_numpy()), shape=(ds_salaries.salary_in_usd.size,))

    def __len__(self):
        return self.salary.size()[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.feature[idx], self.salary[idx]


def getCustomizedData():
    # preprocess
    dataset = DS_SalaryDataset()
    # train test split
    train_count = int(0.7 * len(dataset))
    valid_count = int(0.2 * len(dataset))
    test_count = len(dataset) - train_count - valid_count
    print(train_count, valid_count, test_count)
    trainset, valset, testset = random_split(
        dataset, (train_count, valid_count, test_count), Generator().manual_seed(42))
    datum_size = product(trainset[0][0].size())
    return trainset, valset, testset, datum_size


class TwoLayerNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, init_method: Callable[[torch.Tensor], torch.Tensor], active_func: Callable[[], nn.modules.module.Module],
                 DO: float, if_BN: bool, store_size: int = 1):
        super(TwoLayerNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.if_BN = if_BN
        # dropout
        self.do = nn.Dropout(DO)
        # first layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # batch norm
        self.bn1 = nn.BatchNorm1d(hidden_size)
        # activation
        self.active_func = active_func()
        # second layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # initialize
        for param in self.parameters():
            init_method(param)
        self.storage: deque[List[nn.Parameter]] = deque(maxlen=store_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.do(x)
        out = self.fc1(out)
        if self.if_BN:
            out = self.bn1(out)
        out = self.active_func(out)
        out = self.fc2(out)
        return out


class WD_Regularization(nn.Module):
    def __init__(self):
        super(WD_Regularization, self).__init__()


class L2_Regularization(WD_Regularization):
    def __init__(self, weight_decay: float):
        super(L2_Regularization, self).__init__()
        if weight_decay <= 0:
            raise ValueError("param weight_decay can not <=0!!")
        self.weight_decay = weight_decay

    def forward(self, model: TwoLayerNetwork) -> Union[torch.Tensor, float]:
        reg = 0
        for name, parameter in model.named_parameters():
            if name in ("fc1.weight", "fc2.weight") :
                reg += torch.sum(parameter**2)
        return self.weight_decay * reg


def forward_backward(optimizer: optim.Optimizer, criterion: nn.modules.loss._Loss, wd_reg: Optional[WD_Regularization], model: TwoLayerNetwork, y: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad()
    outputs = model(x)
    loss_all: torch.Tensor = criterion(outputs, y)
    loss_all = loss_all + wd_reg(model)
    loss_all.backward()
    optimizer.step()
    return loss_all, outputs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
    trainset, valset, testset, input_size = getCustomizedData()
    criterion = nn.MSELoss()
    hidden_size = 32
    init: Callable[[torch.Tensor], torch.Tensor] = lambda x: nn.init.xavier_uniform_(
        tensor=x) if len(x.shape) > 1 else x
    active = nn.ReLU
    model = TwoLayerNetwork(input_size, hidden_size, 1,
                            init, active, 0., False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    l2_reg = L2_Regularization(0.0001).to(device)
    for x, y in DataLoader(trainset, batch_size=32, shuffle=True):
        x: torch.Tensor = x.view(-1, model.input_size).to(device)
        y: torch.Tensor = y.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(x)
        loss: torch.Tensor = criterion(outputs, y)
        wd = l2_reg(model)
        loss_all = loss + wd
        if wd == float("inf"):
            print([(n, p) for n, p in model.named_parameters() if "weight" in n])
        loss_all.backward()
        optimizer.step()
