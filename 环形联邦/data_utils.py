import os
from typing import Optional
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, random_split


def minmax_helper(df):
    mmax = max(df.max())
    mmin = min(df.min())
    df = (df - mmin) / (mmax - mmin)
    return df, mmax, mmin


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", config=None):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.num_workers = os.cpu_count() // 2

        self.stat_price = None

        self.batch_size = config["batch_size"]

    def setup(self, stage: Optional[str] = None):
        if self.config["data"] == "all":
            X = pd.read_csv(os.path.join(self.data_dir, "x_3d.csv"))
            Y = pd.read_csv(os.path.join(self.data_dir, "y_3d.csv"))
        else:
            period = self.config["data"]
            X = pd.read_csv(os.path.join(self.data_dir, f"x_3d_{period}.csv"))
            Y = pd.read_csv(os.path.join(self.data_dir, f"y_3d_{period}.csv"))

        X, Y = self.data_preprocess(X, Y)
        if self.config["only_price"]:
            X = X.reshape(-1, 3, 5)[:, :, 0].reshape(-1, 3)


        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=False
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).float()

        self.train = TensorDataset(X_train, y_train)
        self.val = TensorDataset(X_val, y_val)
        self.test = TensorDataset(X_test, y_test)

    @property
    def train_dataset(self):
        return self.train

    @property
    def val_dataset(self):
        return self.val

    @property
    def test_dataset(self):
        return self.test

    def train_dataloader(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        return DataLoader(
            self.train,
            batch_size=batch_size,
            num_workers=0,
            persistent_workers=False,
        )

    def val_dataloader(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        return DataLoader(
            self.val,
            batch_size=batch_size,
            num_workers=0,
            persistent_workers=False,
        )

    def test_dataloader(self, batch_size=None):
        if not batch_size:
            batch_size = 1
        return (
            self.train_dataloader(batch_size),
            self.val_dataloader(batch_size),
            DataLoader(
                self.test,
                batch_size=batch_size,
                num_workers=0,
                persistent_workers=False,
            ),
        )

    def data_preprocess(self, X, Y):
        category_cols = [
            "Port Type",
            "Plug In Event Id",
            "Transaction Date (Pacific Time)",
            "End Date",

        ]

        # data processing

        X = X.drop(columns=category_cols)
        X["Start Date"] = pd.to_datetime(X["Start Date"]).astype('int64') // 10 ** 9  # 转时间戳
        X["Charging Time (hh:mm:ss)"] = X["Charging Time (hh:mm:ss)"].apply(lambda x: sum(int(t) * 60 ** i for i, t in enumerate(reversed(x.split(":")))))




        X, stat_Energy = self.scale(X)
        Y = (Y - stat_Energy["min"]) / (stat_Energy["max"] - stat_Energy["min"])

        X = X.values
        Y = Y.values
        self.stat_price = stat_Energy
        self.config["MAX_PRICE"] = stat_Energy["max"]  # 修复可能的拼写错误
        self.config["MIN_PRICE"] = stat_Energy["min"]
        return X, Y

    @staticmethod
    def scale(df):
        stat_Energy = {"min": None, "max": None}
        df_Energy = df.drop(columns=["Charging Time (hh:mm:ss)","Start Date"]).select_dtypes(
            "float64"
        )

        df_Energy, stat_Energy["max"], stat_Energy["min"] = minmax_helper(df_Energy)
        df[df_Energy.columns] = df_Energy

        df_Charging = df[["Charging Time (hh:mm:ss)", "Start Date"]]
        df_Charging, _, _ = minmax_helper(df_Charging)
        df[df_Charging.columns] = df_Charging

        return df, stat_Energy
