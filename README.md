# Learning-ANN

個人從零實作人工神經網路（ANN）的學習歷程，僅供參考。

## 特色

- **純 NumPy 實作**，不依賴 PyTorch / TensorFlow 等深度學習框架
- 手刻前向傳播、反向傳播（含鏈鎖律推導）與梯度下降
- 提供完整數學筆記與 Jupyter Notebook 練習

## 環境需求

```bash
pip install numpy pandas matplotlib jupyter
```

## 執行方式

```bash
# Jupyter Notebook（探索式學習）
jupyter notebook

# Python 腳本（需在 Python-Example-Code/ 目錄下執行）
cd Python-Example-Code
python simpleTrainExample.py   # 螺旋資料集三分類
python mnistTrain.py           # MNIST 手寫數字辨識
```

## 內容說明

### Notebooks

| 檔案 | 說明 |
|---|---|
| `Simple Model.ipynb` | 最基礎的 ANN 實作入門 |
| `General Model Framework Version 1/2.ipynb` | 通用框架的逐步演進 |
| `Gradient Descent.ipynb` | 梯度下降原理與視覺化 |
| `Numerical Gradient.ipynb` | 數值梯度驗證（對照解析梯度） |
| `MNIST ANN.ipynb` | 使用框架訓練 MNIST |
| `MNIST ANN no package.ipynb` | 純手刻版 MNIST 分類 |

### Python 核心模組（`Python-Example-Code/`）

- **`classAnn.py`**：物件導向神經網路，包含 `Dense` 層與 `NeuralNetwork` 容器
- **`functionsAnn.py`**：損失函數（MSE、BCE、Cross-Entropy）、啟動函數、Mini-Batch 訓練迴圈
- **`simpleTrainExample.py`**：以螺旋資料集示範完整訓練流程
- **`mnistTrain.py`**：以 MNIST CSV 資料集示範圖像分類

### 數學筆記

- **`Baisc ANN Math.md`**：完整推導（符號定義 → 前向傳播 → 損失函數 → 反向傳播四核心公式 → XOR 數值範例）
- **`Baisc ANN Math Easy.md`**：簡化版入門說明

### 資料集（`Data/`）

包含 MNIST、Iris、Heart Failure、Water Quality 等多種公開資料集（CSV / MAT 格式）。
