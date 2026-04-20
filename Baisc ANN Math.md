# 類神經網路的數學推導

> 適合大學一年級程度，從零開始理解前向傳播與反向傳播

---

## 目錄

1. [基本符號定義](#1-基本符號定義)
2. [神經元的數學模型](#2-神經元的數學模型)
3. [激活函數](#3-激活函數)
4. [前向傳播（Forward Propagation）](#4-前向傳播forward-propagation)
5. [損失函數（Loss Function）](#5-損失函數loss-function)
6. [反向傳播（Backpropagation）](#6-反向傳播backpropagation)
7. [參數更新：梯度下降法](#7-參數更新梯度下降法)
8. [完整範例：XOR 問題](#8-完整範例xor-問題)

---

## 1. 基本符號定義

在開始推導之前，先統一符號，避免混亂。

| 符號 | 意義 |
|------|------|
| $L$ | 網路總層數（含輸入層、隱藏層、輸出層） |
| $l$ | 第 $l$ 層，$l = 1, 2, \ldots, L$ |
| $n^{[l]}$ | 第 $l$ 層的神經元數量 |
| $W^{[l]}$ | 第 $l$ 層的**權重矩陣**，大小為 $n^{[l]} \times n^{[l-1]}$ |
| $b^{[l]}$ | 第 $l$ 層的**偏差向量**，大小為 $n^{[l]} \times 1$ |
| $z^{[l]}$ | 第 $l$ 層的**線性組合**（加權和） |
| $a^{[l]}$ | 第 $l$ 層的**激活值**（輸出） |
| $\hat{y}$ | 網路最終預測值，即 $a^{[L]}$ |
| $y$ | 真實標籤 |
| $\mathcal{L}$ | 損失函數 |

> **輸入層** 為第 $1$ 層，所以 $a^{[1]} = x$（輸入資料）。

---

## 2. 神經元的數學模型

### 2.1 單一神經元

想像一個神經元接收 $n$ 個輸入 $x_1, x_2, \ldots, x_n$，每個輸入有對應的權重 $w_1, w_2, \ldots, w_n$，以及一個偏差項 $b$。

**線性組合（Pre-activation）：**

$$z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \sum_{i=1}^{n} w_i x_i + b$$

用向量內積寫得更簡潔：

$$z = \mathbf{w}^T \mathbf{x} + b$$

**激活（Activation）：**

$$a = \sigma(z)$$

其中 $\sigma$ 是激活函數（下一節詳細介紹）。

### 2.2 直覺理解

- **權重 $w_i$**：代表第 $i$ 個輸入的「重要程度」
- **偏差 $b$**：讓神經元可以「平移」其決策邊界
- **激活函數 $\sigma$**：引入非線性，讓網路能學習複雜的模式

---

## 3. 激活函數

激活函數必須是**可微分**的（這是反向傳播的關鍵需求）。

### 3.1 Sigmoid 函數

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**導數推導：**

$$\sigma'(z) = \frac{d}{dz} \left( \frac{1}{1+e^{-z}} \right)$$

利用商法則，令 $u = 1$，$v = 1 + e^{-z}$：

$$\sigma'(z) = \frac{0 \cdot v - u \cdot (-e^{-z})}{v^2} = \frac{e^{-z}}{(1+e^{-z})^2}$$

**整理成美麗的形式：**

$$\sigma'(z) = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z) \cdot \left(1 - \sigma(z)\right)$$

$$\boxed{\sigma'(z) = \sigma(z)\left(1 - \sigma(z)\right)}$$

> 這個結果非常漂亮：只需要知道輸出值 $\sigma(z)$，就能算出導數！

### 3.2 ReLU 函數

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**導數：**

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$

（$z=0$ 處技術上不可微，實作上通常取 $0$）

### 3.3 Tanh 函數

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**導數：**

$$\tanh'(z) = 1 - \tanh^2(z)$$

---

## 4. 前向傳播（Forward Propagation）

前向傳播就是把輸入資料「從左到右」逐層計算，直到得出預測值。

### 4.1 單層的計算

對於第 $l$ 層（$l = 2, 3, \ldots, L$）：

**步驟一：線性組合**

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$

- $W^{[l]}$ 的大小：$n^{[l]} \times n^{[l-1]}$
- $a^{[l-1]}$ 的大小：$n^{[l-1]} \times 1$
- $b^{[l]}$ 的大小：$n^{[l]} \times 1$
- $z^{[l]}$ 的大小：$n^{[l]} \times 1$

**步驟二：激活**

$$a^{[l]} = \sigma^{[l]}\!\left(z^{[l]}\right)$$

（各層可以使用不同的激活函數）

### 4.2 完整流程

以一個三層網路為例（輸入層、一個隱藏層、輸出層）：

```
輸入：a^[1] = x

第 2 層（隱藏層）：
  z^[2] = W^[2] a^[1] + b^[2]
  a^[2] = σ(z^[2])

第 3 層（輸出層）：
  z^[3] = W^[3] a^[2] + b^[3]
  a^[3] = σ(z^[3])

預測值：ŷ = a^[3]
```

### 4.3 矩陣計算的維度檢查

這是避免程式錯誤最重要的習慣！

若輸入 $x$ 有 $n^{[1]} = 3$ 個特徵，隱藏層有 $n^{[2]} = 4$ 個神經元：

$$W^{[2]}: 4 \times 3, \quad a^{[1]}: 3 \times 1 \implies z^{[2]}: 4 \times 1 \checkmark$$

---

## 5. 損失函數（Loss Function）

損失函數衡量預測值 $\hat{y}$ 與真實值 $y$ 的差距。

### 5.1 均方誤差（MSE）——用於迴歸

$$\mathcal{L}(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$$

> 前面的 $\frac{1}{2}$ 是為了讓求導後的結果更乾淨。

**對 $\hat{y}$ 的偏導數：**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y$$

### 5.2 二元交叉熵（BCE）——用於二元分類

$$\mathcal{L}(\hat{y}, y) = -\left[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \right]$$

**對 $\hat{y}$ 的偏導數：**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

---

## 6. 反向傳播（Backpropagation）

反向傳播是整個類神經網路中最核心的算法。它的本質是**微積分的鏈鎖律（Chain Rule）**。

### 6.1 鏈鎖律複習

如果 $y = f(g(x))$，那麼：

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**範例：** 若 $y = (3x+1)^2$

令 $u = 3x+1$，則 $y = u^2$

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 6(3x+1)$$

在神經網路中，損失 $\mathcal{L}$ 是透過多層函數複合而成，我們需要對每個 $W^{[l]}$ 和 $b^{[l]}$ 求偏導數。

### 6.2 定義誤差訊號 $\delta^{[l]}$

為了方便，定義第 $l$ 層的**誤差訊號**：

$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}}$$

這代表「損失對第 $l$ 層線性組合的敏感度」。

### 6.3 輸出層的誤差訊號

對於輸出層（第 $L$ 層），利用鏈鎖律：

$$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial z^{[L]}} = \frac{\partial \mathcal{L}}{\partial a^{[L]}} \cdot \frac{\partial a^{[L]}}{\partial z^{[L]}}$$

$$\boxed{\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial a^{[L]}} \odot \sigma'^{[L]}\!\left(z^{[L]}\right)}$$

其中 $\odot$ 代表**逐元素相乘（element-wise multiplication）**。

**以 MSE + Sigmoid 為例：**

$$\frac{\partial \mathcal{L}}{\partial a^{[L]}} = \hat{y} - y, \quad \sigma'^{[L]}\!\left(z^{[L]}\right) = a^{[L]}\left(1 - a^{[L]}\right)$$

$$\therefore \delta^{[L]} = (\hat{y} - y) \odot \hat{y}(1 - \hat{y})$$

### 6.4 隱藏層的誤差訊號——反向傳遞公式

對於 $l = L-1, L-2, \ldots, 2$，誤差訊號如何從第 $l+1$ 層傳回第 $l$ 層？

利用鏈鎖律展開 $\frac{\partial \mathcal{L}}{\partial z^{[l]}}$：

$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l+1]}} \cdot \frac{\partial z^{[l+1]}}{\partial a^{[l]}} \cdot \frac{\partial a^{[l]}}{\partial z^{[l]}}$$

注意到 $z^{[l+1]} = W^{[l+1]} a^{[l]} + b^{[l+1]}$，所以：

$$\frac{\partial z^{[l+1]}}{\partial a^{[l]}} = W^{[l+1]}$$

因此：

$$\boxed{\delta^{[l]} = \left(W^{[l+1]}\right)^T \delta^{[l+1]} \odot \sigma'^{[l]}\!\left(z^{[l]}\right)}$$

> **直覺理解：** 誤差從後面一層「反向流回來」——乘上轉置權重矩陣，再乘上本層激活函數的導數。

### 6.5 計算對權重和偏差的梯度

有了誤差訊號 $\delta^{[l]}$，就可以算出損失對所有參數的梯度：

**對權重矩陣 $W^{[l]}$ 的梯度：**

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} \cdot \left(a^{[l-1]}\right)^T$$

**對偏差向量 $b^{[l]}$ 的梯度：**

$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}$$

### 6.6 推導 $\frac{\partial \mathcal{L}}{\partial W^{[l]}}$ 的細節

以第 $l$ 層第 $j$ 個神經元為例，其線性組合為：

$$z_j^{[l]} = \sum_k W_{jk}^{[l]} a_k^{[l-1]} + b_j^{[l]}$$

對 $W_{jk}^{[l]}$ 求偏導：

$$\frac{\partial z_j^{[l]}}{\partial W_{jk}^{[l]}} = a_k^{[l-1]}$$

由鏈鎖律：

$$\frac{\partial \mathcal{L}}{\partial W_{jk}^{[l]}} = \frac{\partial \mathcal{L}}{\partial z_j^{[l]}} \cdot \frac{\partial z_j^{[l]}}{\partial W_{jk}^{[l]}} = \delta_j^{[l]} \cdot a_k^{[l-1]}$$

寫成矩陣形式，就是 $\delta^{[l]} \cdot (a^{[l-1]})^T$。✓

---

## 7. 參數更新：梯度下降法

算出梯度之後，就可以更新參數讓損失下降。

### 7.1 梯度下降更新規則

$$W^{[l]} \leftarrow W^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial W^{[l]}}$$

$$b^{[l]} \leftarrow b^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial b^{[l]}}$$

其中 $\alpha > 0$ 是**學習率（Learning Rate）**，控制每次更新的步伐大小。

### 7.2 學習率的影響

| 學習率 | 效果 |
|--------|------|
| 太大 | 步伐過大，可能在最小值附近震盪，甚至發散 |
| 太小 | 收斂極慢，訓練時間過長 |
| 適當 | 穩定收斂至局部最小值 |

### 7.3 Mini-Batch 梯度下降

實作上，通常不用單一樣本，而是取一小批（batch）樣本，計算平均梯度：

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} \approx \frac{1}{m} \sum_{i=1}^{m} \frac{\partial \mathcal{L}^{(i)}}{\partial W^{[l]}}$$

其中 $m$ 是 batch size，$\mathcal{L}^{(i)}$ 是第 $i$ 筆樣本的損失。

---

## 8. 完整範例：XOR 問題

讓我們用一個完整的數值範例，走過一次完整的前向 + 反向傳播。

### 8.1 網路結構

- 輸入層：$n^{[1]} = 2$
- 隱藏層：$n^{[2]} = 2$，使用 Sigmoid
- 輸出層：$n^{[3]} = 1$，使用 Sigmoid
- 損失：MSE

### 8.2 初始參數（假設值）

$$W^{[2]} = \begin{bmatrix} 0.5 & 0.3 \\ -0.2 & 0.8 \end{bmatrix}, \quad b^{[2]} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$

$$W^{[3]} = \begin{bmatrix} 0.6 & -0.4 \end{bmatrix}, \quad b^{[3]} = \begin{bmatrix} 0.2 \end{bmatrix}$$

### 8.3 輸入樣本

$$x = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad y = 1$$

（XOR：$1 \oplus 0 = 1$）

### 8.4 前向傳播計算

**第 2 層（隱藏層）：**

$$z^{[2]} = W^{[2]} x + b^{[2]} = \begin{bmatrix} 0.5(1) + 0.3(0) + 0.1 \\ -0.2(1) + 0.8(0) + (-0.1) \end{bmatrix} = \begin{bmatrix} 0.6 \\ -0.3 \end{bmatrix}$$

$$a^{[2]} = \sigma\!\left(z^{[2]}\right) = \begin{bmatrix} \sigma(0.6) \\ \sigma(-0.3) \end{bmatrix} = \begin{bmatrix} 0.6457 \\ 0.4256 \end{bmatrix}$$

**第 3 層（輸出層）：**

$$z^{[3]} = W^{[3]} a^{[2]} + b^{[3]} = 0.6(0.6457) + (-0.4)(0.4256) + 0.2$$

$$= 0.3874 - 0.1702 + 0.2 = 0.4172$$

$$\hat{y} = a^{[3]} = \sigma(0.4172) = 0.6028$$

### 8.5 計算損失

$$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0.6028 - 1)^2 = \frac{1}{2}(0.1575) = 0.0788$$

### 8.6 反向傳播計算

**輸出層誤差訊號 $\delta^{[3]}$：**

$$\frac{\partial \mathcal{L}}{\partial a^{[3]}} = \hat{y} - y = 0.6028 - 1 = -0.3972$$

$$\sigma'(z^{[3]}) = a^{[3]}(1-a^{[3]}) = 0.6028 \times 0.3972 = 0.2395$$

$$\delta^{[3]} = -0.3972 \times 0.2395 = -0.0951$$

**輸出層梯度：**

$$\frac{\partial \mathcal{L}}{\partial W^{[3]}} = \delta^{[3]} \cdot \left(a^{[2]}\right)^T = [-0.0951] \cdot [0.6457, \ 0.4256]$$

$$= \begin{bmatrix} -0.0614 & -0.0405 \end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial b^{[3]}} = \delta^{[3]} = [-0.0951]$$

**隱藏層誤差訊號 $\delta^{[2]}$：**

$$\delta^{[2]} = \left(W^{[3]}\right)^T \delta^{[3]} \odot \sigma'\!\left(z^{[2]}\right)$$

$$\left(W^{[3]}\right)^T \delta^{[3]} = \begin{bmatrix} 0.6 \\ -0.4 \end{bmatrix} \times (-0.0951) = \begin{bmatrix} -0.0571 \\ 0.0380 \end{bmatrix}$$

$$\sigma'\!\left(z^{[2]}\right) = \begin{bmatrix} 0.6457(1-0.6457) \\ 0.4256(1-0.4256) \end{bmatrix} = \begin{bmatrix} 0.2289 \\ 0.2444 \end{bmatrix}$$

$$\delta^{[2]} = \begin{bmatrix} -0.0571 \\ 0.0380 \end{bmatrix} \odot \begin{bmatrix} 0.2289 \\ 0.2444 \end{bmatrix} = \begin{bmatrix} -0.0131 \\ 0.0093 \end{bmatrix}$$

**隱藏層梯度：**

$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \delta^{[2]} \cdot \left(a^{[1]}\right)^T = \begin{bmatrix} -0.0131 \\ 0.0093 \end{bmatrix} \begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} -0.0131 & 0 \\ 0.0093 & 0 \end{bmatrix}$$

### 8.7 參數更新（學習率 $\alpha = 0.5$）

$$W^{[3]} \leftarrow \begin{bmatrix} 0.6 & -0.4 \end{bmatrix} - 0.5 \begin{bmatrix} -0.0614 & -0.0405 \end{bmatrix} = \begin{bmatrix} 0.6307 & -0.3798 \end{bmatrix}$$

重複以上過程數千次，網路便逐漸學會 XOR 函數。

---

## 總結：反向傳播的四個核心公式

$$\boxed{
\begin{aligned}
&\textbf{BP1（輸出層誤差）：} && \delta^{[L]} = \frac{\partial \mathcal{L}}{\partial a^{[L]}} \odot \sigma'^{[L]}\!\left(z^{[L]}\right) \\[8pt]
&\textbf{BP2（誤差反向傳遞）：} && \delta^{[l]} = \left(W^{[l+1]}\right)^T \delta^{[l+1]} \odot \sigma'^{[l]}\!\left(z^{[l]}\right) \\[8pt]
&\textbf{BP3（權重梯度）：} && \frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} \left(a^{[l-1]}\right)^T \\[8pt]
&\textbf{BP4（偏差梯度）：} && \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}
\end{aligned}
}$$

這四個公式是所有深度學習框架（PyTorch、TensorFlow）的理論基礎。只要理解了這四個公式，你就真正理解了類神經網路是如何「學習」的！

---

*參考概念：微積分鏈鎖律 · 線性代數矩陣運算 · 最佳化理論梯度下降*
