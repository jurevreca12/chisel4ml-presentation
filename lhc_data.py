"Fetches the LHC Jets dataset."
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NumpyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        assert len(x_data) == len(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


def get_lhc_dataset(batch_size=256, split=0.3):
    "Returns Pytorch Dataloaders for training and testing."
    data = fetch_openml('hls4ml_lhc_jets_hlf')
    x, y = data['data'], data['target']
    le = LabelEncoder()
    y = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=split,
        random_state=42
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_train = np.round(x_train * 16)   # "quantize"
    x_test = scaler.transform(x_test)
    x_test = np.round(x_test * 16)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    train_ds = NumpyDataset(x_train, y_train)
    test_ds = NumpyDataset(x_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl