# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 關於此專案

個人從零實作人工神經網路（ANN）的學習歷程，**不依賴深度學習框架**，以純 NumPy 手刻前向傳播、反向傳播與梯度下降。

## 執行方式

```bash
# 執行 Python 腳本範例（需在 Python-Example-Code/ 目錄下執行，因路徑相依）
cd Python-Example-Code
python simpleTrainExample.py   # 螺旋資料集分類
python mnistTrain.py           # MNIST 手寫數字分類

# 啟動 Jupyter Notebook
jupyter notebook
```

依賴套件：`numpy`、`pandas`、`matplotlib`（無 requirements.txt，需手動安裝）。

## 架構說明

### 核心模組（`Python-Example-Code/`）

兩個模組相互配合，是整個專案的核心：

**`classAnn.py`** — 物件導向的神經網路骨架
- `Layer`：抽象基底類別，定義 `forward` / `backward` 介面
- `Dense(Layer)`：全連接層，支援 `relu`、`Sigmoid`、`Softmax` 激活函數；`forward` 時儲存輸入供反向傳播使用
- `NeuralNetwork`：容器，管理多個 `Dense` 層，提供 `forward`、`backward`、`update_parameters`、`reg_loss`（L2 正則化）

**`functionsAnn.py`** — 損失函數、梯度與訓練工具
- 損失函數：`mse_loss_grad`、`binary_cross_entropy_loss_grad`、`cross_entropy_grad_loss`（均同時回傳 loss 與 grad）
- 訓練迴圈：`train_batch`（Mini-Batch 梯度下降，支援 shuffle）
- 工具：`data_iter`（批次產生器）、`gen_spiral_dataset`、`numerical_gradient` / `numerical_gradient_from_df`（數值梯度，用於梯度驗證）

### 訓練流程慣例

```python
f = nn.forward(X)                          # 前向傳播
loss, grad = loss_fn(f, y, ...)            # 計算 loss 與梯度
loss += nn.reg_loss(reg)                   # 加入 L2 正則化
nn.backward(grad, reg)                     # 反向傳播
nn.update_parameters(learning_rate)        # 參數更新
```

損失函數統一回傳 `(loss, grad)` tuple，`backward` 的第一個引數即為輸出層的梯度。

### Softmax 輸出層的特殊處理

`Dense` 層若使用 `Softmax` 激活，其 `dZ_` 方法**不處理** Softmax 的梯度（直接 pass-through），梯度需由 `cross_entropy_grad` 在外部計算並傳入。呼叫 `train_batch` 時以 `softmax_out` 參數控制此行為。

### Notebooks 對應關係

| Notebook | 內容 |
|---|---|
| `Simple Model.ipynb` / `Simple Model-1.ipynb` | 最基礎的 ANN 實作 |
| `General Model Framework Version 1/2.ipynb` | 通用框架演進 |
| `Gradient Descent.ipynb` | 梯度下降原理 |
| `Numerical Gradient.ipynb` | 數值梯度驗證 |
| `MNIST ANN.ipynb` / `MNIST ANN no package.ipynb` | MNIST 分類（有/無套件） |

### 資料集（`Data/`）

`mnistTrain.py` 以相對路徑 `../Data/mnist_train.csv` 讀取，執行時需在 `Python-Example-Code/` 目錄下。

## 數學參考

- `Baisc ANN Math.md`：完整的前向/反向傳播數學推導（含 XOR 數值範例）
- `Baisc ANN Math Easy.md`：簡化版說明
