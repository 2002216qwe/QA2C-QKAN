import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
os.environ["WANDB_DISABLED"] = "true"
from pathlib import Path
from pytorch_lightning import LightningModule
from utils import unscale, count_parameters, load_config
from data_utils import DataModule
from models import XXQLSTM, QLSTM
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

def load_model(
    run,
    exp_id,
    model_name="xx-QLSTM",
    vqc="original",
    encoding="original",
    data="period1",
    hidden_dim=2,
    four_linear_before_vqc=True,
    combine_linear_after_vqc=True,
):
    config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    config["model_name"] = model_name
    config[model_name]["vqc"] = vqc
    config[model_name]["encoding"] = encoding
    config[model_name]["hidden_dim"] = hidden_dim
    config[model_name]["four_linear_before_vqc"] = four_linear_before_vqc
    config[model_name]["combine_linear_after_vqc"] = combine_linear_after_vqc
    config["data"] = data
    exp = f"energy-lab/{model_name}/model-{exp_id}:best"
    # 模拟加载模型，实际应用中需要替换为实际的模型加载逻辑
    model = PricePredictor(config)
    model_path = "/tmp/pycharm_project_507/Quantum-LSTM-master/src/model_checkpoints/lightning_logs/version_8/checkpoints/epoch=299-val_loss=0.15847.ckpt"
    model = PricePredictor.load_from_checkpoint(model_path, config=config)
    return model


def predict(model, period, max_price, min_price):
    config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    config["data"] = period
    data_module = DataModule(
        config=config, data_dir=os.path.join(os.path.dirname(__file__), "../data/")
    )
    data_module.setup()
    predictions = [[], [], []]
    targets = [[], [], []]
    model.eval()
    for i, dataloader in enumerate(data_module.test_dataloader()):
        for X, y in tqdm(dataloader):
            pred = model(X)
            predictions[i].append(pred.item())
            targets[i].append(y.item())
            if i == 0:
                dataset = "training"
            elif i == 1:
                dataset = "validation"
            else:
                dataset = "test"
            pred_unscale = unscale(
                pred.detach().flatten().numpy(),
                max_price,
                min_price,
            )
            print(f"pred/{dataset}: {pred_unscale}")
    mae_fn = lambda a, b: np.abs(np.array(a) - np.array(b)).sum() / len(a)
    mse_fn = lambda a, b: np.square(np.array(a) - np.array(b)).sum() / len(a)
    rmse_fn = lambda a, b: (np.square(np.array(a) - np.array(b)) ** 1 / 2).sum() / len(a)
    for j, dataset in enumerate(["training", "validation", "test"]):
        print(f"{dataset}-MSE: {mse_fn(predictions[j], targets[j])}")
        print(f"{dataset}-MAE: {mae_fn(predictions[j], targets[j])}")
        print(f"{dataset}-RMSE: {rmse_fn(predictions[j], targets[j])}")
    return predictions


class PricePredictor(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.model_name = config["model_name"]
        self.n_qubits = config[self.model_name]["n_qubits"]
        self.input_dim = config[self.model_name]["input_dim"]
        self.hidden_dim = config[self.model_name]["hidden_dim"]
        self.depth = config[self.model_name]["depth"]
        self.backend = config["backend"]
        self.target_size = config["target_size"]
        self.n_timestep = config["n_timestep"]
        self.vqc = config[self.model_name]["vqc"]
        self.dropout_rate = config[self.model_name]["dropout"]

        params_to_save = {
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "model_name": self.model_name,
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "seed": self.config["seed"],
            "device": config["devices"],
            "data": config["data"],
            "vqc": self.vqc,
            "dropout_rate": self.dropout_rate,
        }

        if self.model_name != "LSTM":
            self.diff_method = config[self.model_name]["diff_method"]
            params_to_save["encoding"] = config[self.model_name]["encoding"]

        if self.model_name == "QLSTM":
            self.lstm = QLSTM(config)
        elif self.model_name == "xx-QLSTM":
            self.lstm = XXQLSTM(config)
        else:
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.depth,
                dropout=self.dropout_rate,
            )

        if self.model_name != "QLSTM":
            self.final = nn.Linear(self.hidden_dim, self.target_size)

        params_to_save["n_parameters"] = count_parameters(self)
        self.save_hyperparameters(params_to_save)

    def forward(self, sequence):
        print("Input shape:", sequence.shape)  # 打印输入形状
        sequence = (
            sequence.view(-1, self.n_timestep, self.input_dim)
            .transpose(0, 1)
            .contiguous()
        )
        sequence = (
            sequence.view(-1, self.n_timestep, self.input_dim)
            .transpose(0, 1)
            .contiguous()
        )
        if self.model_name == "QLSTM":
            return self.lstm(sequence)
        lstm_out, _ = self.lstm(sequence)
        return self.final(lstm_out[-1].view(-1, self.hidden_dim))


    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).flatten()
        loss = F.mse_loss(pred, y.flatten())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss, "pred": pred, "target": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).flatten()
        loss = F.mse_loss(pred, y.flatten())
        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True)
        return {"loss": loss, "pred": pred, "target": y}

    def test_step(self, batch, batch_idx, dataset_idx):
        part_index_map = {0: "train", 1: "val", 2: "test"}
        part = part_index_map[dataset_idx]
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)

        # 修复：将张量移动到CPU后再转换为numpy数组
        pred_cpu = pred.cpu().detach().flatten().numpy()
        y_cpu = y.cpu().detach().flatten().numpy()

        pred_unscale = unscale(
            pred_cpu,
            self.config["MAX_PRICE"],
            self.config["MIN_PRICE"],
        )
        target_unscale = unscale(
            y_cpu,
            self.config["MAX_PRICE"],
            self.config["MIN_PRICE"],
        )
        self.log("pred", pred_unscale[0], on_step=True, on_epoch=False)
        self.log("target", target_unscale[0], on_step=True, on_epoch=False)
        self.log("test_loss", loss.item(), prog_bar=True)
        return {"loss": loss, "pred": pred, "target": y}


    def predict_step(self, batch, batch_idx, dataset_idx):
        x, y = batch
        pred = self(x)
        return {"pred": pred, "target": y}

    def configure_optimizers(self):
        optimizer = (
            torch.optim.Adam(self.parameters(), lr=self.lr)
            if self.config["model_name"] != "QLSTM"
            else torch.optim.RMSprop(
                self.parameters(),
                lr=0.1,
                alpha=0.99,
                eps=1e-8,
            )
        )
        return optimizer


class PlotCallback(Callback):
    def __init__(self, datamodule, save_dir="images"):
        super().__init__()
        self.datamodule = datamodule
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.csv_dir = os.path.join(os.path.dirname(save_dir), "csv_data")
        os.makedirs(self.csv_dir, exist_ok=True)

        # 预加载验证数据
        self.val_data = []
        val_loader = datamodule.val_dataloader()
        for batch in val_loader:
            self.val_data.append(batch)

        # 记录训练和验证损失
        self.train_losses = []
        self.val_losses = []
        self.loss_history = []
        self.columns = ["epoch", "train_loss", "val_loss"]

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 20 == 0:  # 每20个epoch保存一次结果
            predictions = []
            targets = []

            # 使用预加载的数据进行预测
            pl_module.eval()
            with torch.no_grad():
                for batch in self.val_data:
                    x, y = batch
                    x = x.to(pl_module.device)
                    y = y.to(pl_module.device)
                    pred = pl_module(x)
                    predictions.append(pred.cpu().numpy().flatten())
                    targets.append(y.cpu().numpy().flatten())

            # 合并并反归一化数据
            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)

            pred_unscaled = unscale(
                predictions,
                pl_module.config["MAX_PRICE"],
                pl_module.config["MIN_PRICE"],
            )
            target_unscaled = unscale(
                targets,
                pl_module.config["MAX_PRICE"],
                pl_module.config["MIN_PRICE"],
            )

            # 确保所有数组长度一致
            min_length = min(len(pred_unscaled), len(target_unscaled))
            time_steps = np.arange(min_length)

            # 创建DataFrame并保存
            data_df = pd.DataFrame({
                "time_step": time_steps,
                "true_value": target_unscaled[:min_length],
                "predicted_value": pred_unscaled[:min_length],
                "epoch": trainer.current_epoch + 1
            })

            csv_path = os.path.join(
                self.save_dir,
                f"epoch_{trainer.current_epoch + 1}_data.csv"
            )
            data_df.to_csv(csv_path, index=False)

            # 绘制预测结果图
            plt.figure(figsize=(10, 6))
            plt.plot(target_unscaled[:min_length], label="Ground Truth", alpha=0.7)
            plt.plot(pred_unscaled[:min_length], label="Prediction", linestyle="--", alpha=0.7)
            plt.title(f"Epoch {trainer.current_epoch + 1}")
            plt.xlabel("Time Step")
            plt.ylabel("Nominal Output Power")
            plt.legend()

            plot_path = os.path.join(
                self.save_dir,
                f"epoch_{trainer.current_epoch + 1}_plot.png"
            )
            plt.savefig(plot_path)
            plt.close()

            # 记录损失
            current_train_loss = trainer.callback_metrics.get("train_loss")
            current_val_loss = trainer.callback_metrics.get("val_loss")

            if current_train_loss is not None:
                self.train_losses.append(float(current_train_loss))
            if current_val_loss is not None:
                self.val_losses.append(float(current_val_loss))

            self.loss_history.append({
                "epoch": trainer.current_epoch + 1,
                "train_loss": float(current_train_loss) if current_train_loss else None,
                "val_loss": float(current_val_loss) if current_val_loss else None
            })

    def on_train_end(self, trainer, pl_module):
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)

        plt.plot(epochs, self.train_losses, label='Training Loss')
        if len(self.val_losses) > 0:
            plt.plot(epochs, self.val_losses, label='Validation Loss')

        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        loss_plot_path = os.path.join(self.save_dir, "loss_curve.png")
        plt.savefig(loss_plot_path)
        plt.close()

        # 保存损失历史记录
        df = pd.DataFrame(self.loss_history)
        csv_path = os.path.join(self.save_dir, "loss_history.csv")
        df.to_csv(csv_path, index=False)
        print(f"Loss history saved to {csv_path}")

