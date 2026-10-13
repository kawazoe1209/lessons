# 第3章：学習の仕組み - モデルはどうやって賢くなるのか

> **講義時間目安**: 約40分  

> **ゴール**: 「PyTorch の学習ループ 5 行が、裏で動く数理（損失・勾配・逆伝播・更新）と直結して頭に浮かぶ状態になる」

---

## 【講義の全体タイムテーブル（40分）】

| セクション | 時間目安 | 講義の狙い・メリハリ |
|---|---|---|
| **前提: なぜ W と b を最適化するのか** | 5分 | 2章との接続。動かせるツマミが W と b だけであることを確認 |
| **3.1 損失関数 (Loss Function)** | 7分 | 「ズレ」の数値化。なぜ正解率ではなく微分可能な関数なのか |
| **3.2 勾配降下法 (Gradient Descent)** | 10分 | 傾きの逆向きに進むルール。学習率の罠とミニバッチの意義 |
| **3.3 誤差逆伝播法 (Backpropagation)** | 11分 | 連鎖律による全勾配の一括計算。PyTorch 学習ループ 5 行の解剖 |
| **3.4 最適化アルゴリズム (Optimizer)** | 5分 | 慣性＋歩幅調整＝Adam。実務での選定指針 |
| **まとめ & 第4章への接続** | 2分 | 学習の本質と過学習への伏線 |

---

## 【前提】なぜ $W$ と $b$ を最適化するのか？（5分）

### 1. 第2章の復習：モデル内部の計算式

第2章で学んだ全結合層の順伝播の基本式は次の通りです。

$$
\mathbf{y} = f(W\mathbf{x} + \mathbf{b})
$$

この数式を構成する要素を、「人間やモデルが値を自由に変更できるかどうか」で分類します。

| 記号 | 要素名 | 変更可否 | 役割と意味 |
|---|---|:---:|---|
| $\mathbf{x}$ | **入力データ** | **不可** | 画像のピクセル値、顧客データなど。現実の観測値であり固定。 |
| $\mathbf{y}_{\text{true}}$ | **正解ラベル** | **不可** | データセットに付与された正解（教師データ）。固定。 |
| $f$ | **活性化関数** | **不可** | ReLU や Sigmoid など。モデルの設計段階で固定した関数。 |
| $W$ | **重み行列 (Weight)** | **可** | 各入力特徴量の重要度（結合の強さ・境界線の傾き） |
| $\mathbf{b}$ | **バイアス (Bias)** | **可** | ニューロンの発火しやすさ（判定の閾値・境界線の位置） |

> **前提の最重要ポイント**  

> 入力データも正解も変えられません。モデルの予測精度をコントロールするために**動かせる変数は $W$ と $\mathbf{b}$ の2つしか存在しません**。

---

### 2. なぜ $b$（バイアス）が不可欠なのか？

数式の上で $W$ だけではなく $\mathbf{b}$ が必要な理由は、第2章のパーセプトロンの決定境界を幾何学的に振り返ると明確になります。

* **$W$ のみの場合 ($\mathbf{b} = \mathbf{0}$)**：

  決定境界（分離直線）は**必ず原点 $(0, 0)$ を通る直線**に縛られます。

* **$\mathbf{b}$ がある場合**：

  境界線を**上下左右に平行移動**させることができます。

第2章の論理ゲートの例を思い出してください。同じ重み $W = [0.5, 0.5]$ でも、

* $b = -0.7$（発火ハードル高） $\rightarrow$ **AND ゲート**

* $b = -0.2$（発火ハードル低） $\rightarrow$ **OR ゲート**

傾き（$W$）だけでなく、発火のハードル（$\mathbf{b}$）を調整して初めて、任意のデータ分布に対して正しい境界線を引くことができます。

---

<!-- ========================================== -->

<!-- スライド画像 1：W と b の幾何学的役割 -->

<!-- ========================================== -->

![スライド画像: Wとbの役割と決定境界](slides/01_w_and_b_roles.png)

> **【スライド 1 生成プロンプト】**

> - **用途**: 決定境界における $W$（傾き）と $b$（切片/位置）の幾何学的役割を直感的に解説するスライド

> - **プロンプト（英語）**:

>   `A clean technical slide for machine learning presentation. 2D coordinate plane showing linear classification. Left side: lines rotating showing weight W changing slope while passing through origin (0,0). Right side: lines shifting parallel up and down showing bias b changing position without changing slope. Minimalist, modern dark mode UI, clear vector diagram style, white and cyan contrast, 16:9 ratio.`

> - **スライド記載テキスト（指示）**:

>   - 左図: 「$W$ (重み) = 境界線の傾き（どの入力を重視するか）」

>   - 右図: 「$b$ (バイアス) = 境界線の平行移動（発火のハードルの高さ）」

---

### 3. 「学習」の定義

* **初期状態**：

  $W$ と $\mathbf{b}$ には**ランダムな初期値**が入っています。そのため、学習前のモデルは完全にデタラメな予測しか出力しません。

* **学習（Training）とは**：

  正解データとのズレが最小になるよう、**$W$ と $\mathbf{b}$ の数値をデータに基づいて自動調整し、最適な値へと収束させるプロセス**を指します。

---

### 4. コンピュータが賢くなる実行サイクル（4ステップ）

プログラム内部（PyTorch 等）では、データ1バッチごとに以下の4ステップを高速に繰り返します。

```
 ┌─────────────────────────────────────────────────────────────┐
 │ ① 順伝播 (Forward)                                          │
 │    現在の W, b を使って入力 x から予測 y_pred を計算（第2章）│
 │      ↓                                                      │
 │ ② 損失計算 (Loss)                                           │
 │    正解 y_true とのズレを「損失関数」で1つの数値 L に要約   │
 │      ↓                                                      │
 │ ③ 勾配計算 (Backward)                                       │
 │    「誤差逆伝播法」により、全 W・全 b の傾き (∂L/∂W, ∂L/∂b) を計算 │
 │      ↓                                                      │
 │ ④ パラメータ更新 (Step)                                     │
 │    求めた傾きをもとに、「最適化アルゴリズム」が W, b を書き換える │
 │      ↓                                                      │
 │ └──────┴─── これを数千〜数万回繰り返して W, b を最適化する ───┘
```

本章では、まずステップ②の**「正解とのズレをどうやって数値化するのか（損失関数）」**から見ていきます。

---

<!-- ========================================== -->

<!-- スライド画像 2：深層学習の4ステップ実行サイクル -->

<!-- ========================================== -->

![スライド画像: 深層学習の学習サイクル](slides/02_training_cycle.png)

> **【スライド 2 生成プロンプト】**

> - **用途**: 学習の4ステップ（Forward $\rightarrow$ Loss $\rightarrow$ Backward $\rightarrow$ Step）の循環サイクルを示すスライド

> - **プロンプト（英語）**:

>   `Modern infographic flowchart diagram of deep learning training loop. Circular 4-step process: 1. Forward Propagation (Compute prediction), 2. Loss Calculation (Evaluate error L), 3. Backward Propagation (Compute gradients via chain rule), 4. Parameter Update (Optimizer adjusts W and b). Flat minimal tech design, vibrant neon blue and orange accents on dark navy background, clean typography, 16:9 ratio.`

> - **スライド記載テキスト（指示）**:

>   1. Forward（順伝播: 予測を計算する）

>   2. Loss（損失計算: ズレを数値化する）★本セクション

>   3. Backward（誤差逆伝播: 傾きを測る）

>   4. Step（パラメータ更新: $W, b$ を書き換える）

---

## 3.1 損失関数（Loss Function）（7分）

### 1. なぜ「損失関数」が必要なのか？

「モデルの予測精度を上げる」「賢くする」という曖昧な指示のままでは、コンピュータは計算できません。

予測値と正解値のズレを計算し、**「このスカラー値 $L$ を 0 に近づければ良い」という明確な数学の最適化問題（関数の最小値探索）**に落とし込むために損失関数を定義します。

$$
L(\mathbf{y}_{\text{pred}}, \mathbf{y}_{\text{true}}) \ge 0 \quad (\text{0 が理想})
$$

---

### 2. なぜ評価指標の「正解率（Accuracy）」を直接最適化しないのか？

受講者から最も出やすい質問です。「分類問題なら、正解率が最大になるように直接プログラムを書けばいいのでは？」と考えがちですが、数学的に不可能です。

* **正解率（Accuracy）の弱点**：

  正解率は「何問当たったか」の件数割合であるため、値が**階段状（不連続）**にしか変化しません。

  パラメータ $W, b$ をわずかに動かしても、正解件数が変わらなければ正解率は 1 ピクセルも動きません。つまり**ほぼ全域で微分（傾き）が 0** になります。傾きが 0 だと、「パラメータを増やすべきか減らすべきか」の手がかりが一切得られません。

* **損失関数（Loss）の要件**：

  パラメータのわずかな変化に対して、値が**連続的・なめらかに変化し、全域で微分可能（傾きが存在する）**である必要があります。

> **用語の使い分け**  

> ** **損失関数（Loss）**：モデルがパラメータを更新するために**内部**で使う（微分可能・なめらか）*

*> ** **正解率（Accuracy）**：人間がモデルの出来栄えを評価するために**外部**から見る（直感的・微分不可）

---

<!-- ========================================== -->

<!-- スライド画像 3：正解率 vs 損失関数の勾配比較 -->

<!-- ========================================== -->

![スライド画像: 正解率と損失関数の比較](slides/03_accuracy_vs_loss.png)

> **【スライド 3 生成プロンプト】**

> - **用途**: 正解率（階段状・微分不可）と損失関数（なめらかなすり鉢状・勾配降下可能）の対比スライド

> - **プロンプト（英語）**:

>   `Side-by-side technical comparison chart on dark theme. Left: Accuracy function showing a flat, step-like staircase graph with slope=0 annotations, indicating gradient descent is impossible. Right: Loss function showing a smooth convex U-shaped curve with tangent arrows indicating continuous slope and clear descent direction toward minimum. Modern tech diagram, presentation slide layout, high contrast, 16:9 ratio.`

> - **スライド記載テキスト（指示）**:

>   - 左: 「正解率（階段状）：傾きが 0 なので、どちらへ動かせば改善するか分からない」

>   - 右: 「損失関数（なめらか）：傾き（微分）が計算でき、谷底への方向が分かる」

---

### 3. 回帰タスクの損失：平均二乗誤差（MSE: Mean Squared Error）

数値を直接予測するタスク（売上予測、株価予測、気温予測など）では、**平均二乗誤差（MSE）**が基本となります。

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})^2
$$

#### なぜ二乗するのか？（3つの理由）

1. **正負の相殺を防ぐ**：

   $+2$ の誤差と $-2$ の誤差を足して「合計誤差 0」と判定されるのを防ぐ。

2. **数学的に扱いやすい（全域で微分可能）**：

   絶対値 $|y_{\text{pred}} - y_{\text{true}}|$（MAE）は原点で折れ曲がり微分不能になりますが、二乗関数 $(x)^2$ は原点も含めてなめらかに微分できます。

3. **大きなズレをより重く罰する**：

   ズレが 2 倍になれば損失は 4 倍、3 倍になれば 9 倍になります。モデルに対して「小さな誤差をちまちま直すより、致命的な大外れを最優先で修正しろ」という強い圧力をかけます。

```python
import numpy as np
# 4件のデータに対する予測値と正解値
y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.8, 5.4, 3.0, 6.2])
# 各サンプルの誤差: [-0.2, +0.4, +0.5, -0.8]
# 二乗した誤差:     [0.04, 0.16, 0.25, 0.64]
mse = np.mean((y_pred - y_true) ** 2)
print("MSE:", round(mse, 4))  # 0.2725
```

---

### 4. 分類タスクの損失：交差エントロピー誤差（Cross-Entropy Loss）

クラス分類タスク（画像判定、感情分析など）では、モデルの出力は「各クラスである確率」になります。

正解クラスのインデックスを $k$、モデルが出力した正解クラスの予測確率を $p_k$ としたとき、1サンプルの交差エントロピー損失は次の式で計算されます。

$$
L = -\log(p_k)
$$

#### なぜ $-\log(p)$ なのか？

確率 $p_k$ は $0 \le p_k \le 1$ の範囲を取ります。このとき関数 $L = -\log(p)$ は以下の振る舞いをします。

| 正解クラスへの予測確率 $p_k$ | 損失 $L = -\log(p_k)$ | 判定とモデルへのフィードバック |
|---|---|---|
| **$0.99$** | **$0.01$** | ほぼ満点。ペナルティはほぼゼロ。 |
| **$0.70$** | **$0.36$** | 合格点だが、もう少し改善の余地あり。 |
| **$0.50$** | **$0.69$** | 当てずっぽう。相応のペナルティ。 |
| **$0.10$** | **$2.30$** | 大きく外した。重い修正圧力をかける。 |
| **$0.01$** | **$4.61$** | **強い確信を持って間違えた $\rightarrow$ 損失が急激に爆発** |

* $p \rightarrow 1$（完全正解）のとき、$\log(1) = 0$ なので**損失は 0**

* $p \rightarrow 0$（自信満々に不正解）のとき、**損失は $+\infty$ へ急上昇**

> **本質**  

> 交差エントロピーは**「正解に対してどれだけ確信を持っていたか」の評価スコア**です。外したときの確信度が高い（正解確率が低い）ほど、極端に大きなペナルティを与えてパラメータを急修正させます。

```python
import numpy as np
def cross_entropy(probs, label_idx):
    return -np.log(probs[label_idx] + 1e-9)   # 1e-9 は log(0) のエラー回避
# 3クラス分類（正解はクラス0）
probs_good = np.array([0.7, 0.2, 0.1])       # クラス0を高確率で予測
probs_bad  = np.array([0.1, 0.2, 0.7])       # クラス0を低確率で見積もった
print(round(cross_entropy(probs_good, 0), 3))   # 0.357
print(round(cross_entropy(probs_bad,  0), 3))   # 2.303（ペナルティ激増）
```

---

<!-- ========================================== -->

<!-- スライド画像 4：交差エントロピー損失の関数曲線 -->

<!-- ========================================== -->

![スライド画像: 交差エントロピー損失曲線](slides/04_cross_entropy_curve.png)

> **【スライド 4 生成プロンプト】**

> - **用途**: $L = -\log(p)$ のグラフ形状とペナルティの特性を直感的に示すスライド

> - **プロンプト（英語）**:

>   `Mathematical coordinate graph of function y = -log(x) for x between 0 and 1. X-axis labeled "Predicted Probability for Correct Class p (0 to 1.0)". Y-axis labeled "Loss L". The curve approaches infinity near x=0 with a warning red area labeled "Severe Penalty", and smoothly drops to 0 at x=1 labeled "Zero Loss". Clean infographic aesthetic, dark blue background, crisp glowing lines, 16:9 ratio.`

> - **スライド記載テキスト（指示）**:

>   - $x$軸: 正解クラスの予測確率 $p$（0.0 〜 1.0）

>   - $y$軸: 損失 $L$

>   - 注釈: 「正解確率が 1 に近づくほど損失は 0、0 に近づくほどペナルティが急上昇」

---

### 5. PyTorch での対応表と「二重適用の罠」

実務におけるタスク、モデル出力層、損失関数の対応関係です。

| タスク | モデル最終層（活性化関数） | PyTorch の損失関数クラス | 特記事項 |
|---|---|---|---|
| **回帰** | **なし（恒等関数）** | `nn.MSELoss()` | 予測値をそのまま二乗誤差で評価 |
| **2値分類** | **なし（ロジットを出力）** | `nn.BCEWithLogitsLoss()` | **内部で Sigmoid を自動適用** |
| **多クラス分類** | **なし（ロジットを出力）** | `nn.CrossEntropyLoss()` | **内部で Softmax を自動適用** |

#### ⚠️ 初心者が最もハマる「二重適用バグ」

PyTorch の `CrossEntropyLoss` と `BCEWithLogitsLoss` は、**計算の数値安定性（$\log(0)$ による発散や桁あふれの防止）のため、内部に Softmax / Sigmoid の計算を包含**しています。

もしモデル側の出力層に `nn.Softmax()` を書いてしまうと、

$$
\text{入力} \rightarrow \text{Softmax} \rightarrow \text{Softmax(内部)} \rightarrow \text{Loss}
$$

と**2回適用されてしまい、勾配が正しく伝わらず学習が壊れます**。

> **現場の鉄則**  

> **「PyTorch のニューラルネットワークモデルは、常に生のスコア（Logits）を出力させる」**

```python
import torch
import torch.nn as nn
# 3クラス分類の例
# モデルからは Softmax を通す前の「生のスコア（Logits）」を出力
logits = torch.tensor([
    [ 2.0,  1.0,  0.1],  # サンプル1（正解: クラス0）
    [ 0.5,  0.2,  3.0]   # サンプル2（正解: クラス2）
])
# 正解ラベル（0, 1, 2 のクラス番号）
labels = torch.tensor([0, 2])
# 損失関数の定義と計算
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
print("バッチの平均損失:", round(loss.item(), 4))  # 0.2753
```

---

## 3.2 勾配降下法（Gradient Descent）（10分）

### 1. 損失を小さくする方向へパラメータを動かす基本原理

損失 $L$ を計算できるようになりました。では、$W$ と $\mathbf{b}$ をどちら向きに動かせば $L$ は下がるのでしょうか？

パラメータが1つだけ（$w$）のケースを考えます。

ある地点における**微分（勾配: Gradient） $\frac{\partial L}{\partial w}$** は、「$w$ を微小に増やしたとき、損失 $L$ が増えるか減るか」を表す傾きです。

* **傾きが正（右肩上がり）の場合**：$w$ を**減らせば**、損失 $L$ は下がる（左へ進む）

* **傾きが負（右肩下がり）の場合**：$w$ を**増やせば**、損失 $L$ は下がる（右へ進む）

つまり、**「傾き（勾配）と逆の向き（マイナス方向）へ、適切な歩幅で進む」**ことで、損失を減らすことを目指します。歩幅が大きすぎると損失が減らない場合もあります。これが**勾配降下法**の根本原理です。

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$

$$
\mathbf{b} \leftarrow \mathbf{b} - \eta \frac{\partial L}{\partial \mathbf{b}}
$$

ここで $\eta$（イータ）は**学習率（Learning Rate）**と呼ばれ、一度に踏み出す「歩幅」を決定します。

比喩で言えば、**濃い霧の中で山を下る**作業です。全体の地形は見えませんが、足元の地面の傾きは分かります。「上り坂と逆の下り坂の向きへ一歩踏み出す」。それを繰り返せば谷底へ到達できます。

---

<!-- ========================================== -->

<!-- スライド画像 5：勾配降下法の地形と更新ベクトル -->

<!-- ========================================== -->

![スライド画像: 勾配降下法の地形図](slides/05_gradient_descent_valley.png)

> **【スライド 5 生成プロンプト】**

> - **用途**: 3Dまたは2Dのすり鉢状の谷で、傾きの逆向きにボールが転がり落ちていく様子を示すスライド

> - **プロンプト（英語）**:

>   `3D visualization of a smooth convex loss function surface shaped like a bowl. A glowing point representing current parameters moves downward along contour lines toward the global minimum at the center. Red vectors show positive gradient direction pointing uphill, and green arrows show update step pointing downhill in opposite direction. Dark sleek tech aesthetic, glowing data points, 16:9 ratio.`

> - **スライド記載テキスト**:

>   - 式: $W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}$

>   - 矢印: 「勾配（上り坂）の逆向きへ一歩踏み出す」

---

```python
def loss(w):
    return (w - 3) ** 2
def grad(w):            # 損失の微分（傾き）： 2(w - 3)
    return 2 * (w - 3)
w = 0.0                 # 初期値（ランダムな重みのつもり）
lr = 0.1                # 学習率（一歩の歩幅）
for step in range(1, 21):
    g = grad(w)
    w = w - lr * g      # ← これが勾配降下法の全て
    if step % 5 == 0:
        print(f"step {step:2d}: w = {w:.4f}, loss = {loss(w):.6f}")
# step  5: w = 2.0170, loss = 0.966368
# step 10: w = 2.6779, loss = 0.103763
# step 15: w = 2.8944, loss = 0.011141
# step 20: w = 2.9654, loss = 0.001196 (目標の 3.0 へ収束)
```

実際の深層学習も原理は全く同じです。違いは、動かすパラメータが1個ではなく数百万〜数千億個あることだけです。

### 2. 学習率（Learning Rate）の重要性と影響

学習率 $\eta$ は、深層学習において最も影響の大きいハイパーパラメータです。

| 学習率の大きさ | 挙動と症状 | 結果 |
|---|---|---|
| **小さすぎる（例: 1e-6）** | 歩幅が狭すぎて、ほとんど動かない | 学習に膨大な時間がかかり、途中で止まる |
| **適切（例: 1e-3）** | 安定して谷底に向かって損失が減少する | 良好なパラメータに収束する |
| **大きすぎる（例: 1.0）** | 谷の反対側の崖へ飛び越えてしまい、振動・発散する | 損失が乱高下し、最終的に `NaN` になる |

> **実務の出発点**  

> 後述の Adam を使う場合、**`**lr = 1e-3 (0.001)`** が標準の初期値です。学習が不安定なら 1 桁下げ、遅すぎるなら 1 桁上げる、という進め方をします。

---

<!-- ========================================== -->

<!-- スライド画像 6：学習率の大小による3つの挙動比較 -->

<!-- ========================================== -->

![スライド画像: 学習率の比較](slides/06_learning_rate_comparison.png)

> **【スライド 6 生成プロンプト】**

> - **用途**: 学習率が「小さすぎる」「適切」「大きすぎる」場合の3パターンの挙動比較スライド

> - **プロンプト（英語）**:

>   `Three side-by-side plots of loss curves in a U-shaped valley. Left: Learning rate too small, tiny slow steps creeping down. Middle: Learning rate optimal, smooth steady descent reaching minimum in few steps. Right: Learning rate too large, drastic overshoot oscillating wildly and escaping the valley (divergence). Dark tech background, minimalist UI chart design, 16:9 ratio.`

> - **スライド記載テキスト**:

>   - 左: 「小さすぎる（進まない・停滞）」

>   - 中: 「適切（安定して最小値へ収束）」

>   - 右: 「大きすぎる（飛び越えて発散・NaN）」

---

### 3. バッチ / ミニバッチ / 確率的勾配降下法（SGD）

「1回のパラメータ更新に何件のデータを使って勾配を計算するか」で、3つの手法に分かれます。

| 手法 | 1回の更新に使うデータ数 | メリット | デメリット |
|---|---|---|---|
| **バッチ勾配降下法** | **全データ** (Full Dataset) | 勾配が正確で安定して進む | 1回の計算が極めて重く、全データが GPU メモリに乗らない |
| **確率的勾配降下法 (純SGD)** | **1件** | 計算が超高速。メモリを食わない | 方向がブレまくり、挙動が非常に不安定 |
| **ミニバッチ勾配降下法** | **32〜256件程度** | **GPU の並列計算を最大活用でき、高速かつ安定** | バッチサイズのチューニングが必要 |

> **現代の標準**  

> 現代の深層学習で単に「SGD」や「学習」と呼ぶ場合、**99% は「ミニバッチ勾配降下法」を指します**。なお、更新方向の適度なブレは、浅い谷（局所解）から抜け出しやすくなるメリットもあります。

---

<!-- ========================================== -->

<!-- スライド画像 7：バッチ vs ミニバッチ vs SGD の軌跡 -->

<!-- ========================================== -->

![スライド画像: バッチ方式による軌跡の比較](slides/07_batch_vs_minibatch.png)

> **【スライド 7 生成プロンプト】**

> - **用途**: 等高線マップ上で、バッチ（直線的だが重い）、SGD（激しく蛇行）、ミニバッチ（適度なブレを伴い最速で収束）の移動軌跡を対比するスライド

> - **プロンプト（英語）**:

>   `Contour plot of a 2D loss surface showing optimization paths from a starting point to the center minimum. Three paths compared: 1. Full Batch (smooth, straight, slow line), 2. Pure SGD (wild zigzag erratic noisy path), 3. Mini-batch (slightly noisy but fast, smooth curved path directly hitting center). Dark blue background, vibrant neon colored path lines, tech presentation slide, 16:9 ratio.`

> - **スライド記載テキスト**:

>   - バッチ: 「滑らかだが重い（全データ）」

>   - 純SGD: 「ブレが激しい（1データ）」

>   - ミニバッチ: **「実務のデファクト標準（GPU並列を最大化）」**

---

## 3.3 誤差逆伝播法（Backpropagation）（11分）

### 1. なぜ「逆向き」に計算するのか？（計算量の壁）

勾配降下法を実行するには、すべてのパラメータについて勾配 $\frac{\partial L}{\partial W}, \frac{\partial L}{\partial \mathbf{b}}$ を知る必要があります。

もしネットワークにパラメータが **1億個** あったとします。

* **素朴な方法（数値微分）**：

  $W_1$ を 0.0001 だけ動かして順伝播を再実行 $\rightarrow$ ズレを見る。  

  $W_2$ を 0.0001 だけ動かして順伝播を再実行 $\rightarrow$ ズレを見る。  

  これをパラメータごとに繰り返す必要があり、パラメータ数が多い場合は非常に高コストになります。

* **誤差逆伝播法（バックプロパゲーション）**：

  出力から入力へ向かって**微分の掛け算（連鎖律）を逆向きに伝える**ことで、1回の逆伝播処理で全パラメータの勾配を効率よく計算できます。計算量はネットワークの演算量に依存し、定数時間という意味ではありません。

---

<!-- ========================================== -->

<!-- スライド画像 8：数値微分 vs 誤差逆伝播法の計算量比較 -->

<!-- ========================================== -->

![スライド画像: 数値微分と逆伝播の計算量](slides/08_numerical_vs_backprop.png)

> **【スライド 8 生成プロンプト】**

> - **用途**: パラメータ数 N に対して順伝播を N 回繰り返す数値微分と、1回の逆伝播で全員の勾配が求まる誤差逆伝播法の計算コスト対比

> - **プロンプト（英語）**:

>   `Infographic comparing computation efficiency. Top panel: "Numerical Differentiation" showing 100 million repeated forward computations with a red bottleneck warning "Complexity: O(N) - Impossible". Bottom panel: "Backpropagation" showing single forward pass followed by single backward reverse gradient flow with green checkmark "Complexity: O(1) - Instantaneous". High-tech schematic diagram, dark mode UI, 16:9 ratio.`

> - **スライド記載テキスト**:

>   - 上: 「数値微分：パラメータごとに順伝播（1億回走査 $\rightarrow$ 計算破綻）」

>   - 下: 「誤差逆伝播：逆向きに1回流すだけ（全パラメータ一括計算 $\rightarrow$ 実用化）」

---

### 2. 連鎖律（Chain Rule）と計算グラフ

誤差逆伝播法の正体は、高校数学で習う**合成関数の微分（連鎖律）**です。

$$
y = f(u), \quad u = g(x) \quad \Longrightarrow \quad \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

ニューラルネットワークは関数の入れ子構造なので、出力側から微分を掛け算しながら遡っていけば、すべての層の勾配が芋づる式に求まります。

PyTorch などのフレームワークは、順伝播を行う際にメモリ上に演算の履歴である**計算グラフ（DAG: 有向非巡回グラフ）**を構築し、自動で微分（**autograd**）を実行します。

```
 [順伝播：計算グラフの構築]
  x ──┐
      ├─[ × ]── u ──┐
  W ──┘             ├─[ + ]── z ──[ ReLU ]── a ──[ 損失関数 ]── L
                b ──┘
 [逆伝播：勾配のバケツリレー]
  ∂L/∂W ←─── (× x) ←─── ∂L/∂u ←── (× 1) ←── ∂L/∂z ←── (× f') ←── ∂L/∂a ←── (∂L/∂L=1)
```

```python
import torch
# requires_grad=True で「この変数の傾きを後で追跡する」と宣言
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)
z = w * x + b            # 順伝播（計算グラフが構築される）
loss = (z - 10.0) ** 2   # 正解 10.0 との二乗誤差
loss.backward()          # ← 逆伝播！ 裏でグラフを遡り、全パラメータの勾配を一括計算
print(loss.item())       # 9.0   （z=7 なので (7-10)^2 = 9）
print(w.grad)            # -18.0 （w の足元の傾きが記録された）
print(b.grad)            # -6.0  （b の足元の傾きが記録された）
```

`loss.backward()` を呼ぶと、各変数の `.grad` という箱に傾きが自動で書き込まれます。**ただし、この時点ではまだ傾きを測っただけで、パラメータ自体は1ミリも動いていません**。

---

<!-- ========================================== -->

<!-- スライド画像 9：計算グラフと連鎖律による逆伝播 -->

<!-- ========================================== -->

![スライド画像: 計算グラフと連鎖律](slides/09_computational_graph.png)

> **【スライド 9 生成プロンプト】**

> - **用途**: 計算グラフ上で順伝播のフォワードフローと、逆伝播の勾配リレー（連鎖律）を視覚化したスライド

> - **プロンプト（英語）**:

>   `Detailed computational graph diagram for neural network backpropagation. Left to right: nodes x, W, b connecting to multiplication, addition, activation function, and Loss L via blue forward arrows. Right to left: bright glowing orange reverse arrows showing gradient flow backward, labeling partial derivatives multiplied via chain rule. Clean dataflow diagram, dark navy background, modern UI presentation, 16:9 ratio.`

> - **スライド記載テキスト**:

>   - 青矢印（順伝播）: 「データと重みを掛け合わせ、損失 $L$ を算出」

>   - 橙矢印（逆伝播）: 「連鎖律（掛け算）で出力から入力へ傾きを逆流」

---

### 3. 【最重要】PyTorch 学習ループ 5 行の解剖

これまでの理論が、PyTorch の学習コード 5 行に完全に凝縮されています。

```python
import torch
import torch.nn as nn
# XOR 回路のデータ
X = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])
model = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 1),
)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for epoch in range(1, 501):
    optimizer.zero_grad()        # ① 前回の傾きメモを白紙にリセット
    pred = model(X)              # ② 順伝播：現在の W, b で予測を出す
    loss = criterion(pred, y)    # ③ 損失計算：正解とのズレを数値化
    loss.backward()              # ④ 逆伝播：全パラメータの傾きを一斉に測る
    optimizer.step()             # ⑤ 更新：傾きをもとに W, b を実際に動かす！
    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss = {loss.item():.4f}")
print(torch.sigmoid(model(X)).round().ravel())   # tensor([0., 1., 1., 0.])
```

第2章で「単層では解けない」と述べた XOR が、隠れ層を持つネットワークで解けていることを確認してください。

#### ⚠️ 最頻出バグ：`optimizer.zero_grad()` を忘れるとどうなるか？

PyTorch は `.backward()` を呼ぶたびに、傾きを上書きするのではなく**前回値に足し算（累積）する**仕様になっています。リセットを忘れると傾きが雪だるま式に巨大化し、学習が一瞬で大爆発（`NaN`）します。初心者が最もよくやるバグの筆頭です。

---

<!-- ========================================== -->

<!-- スライド画像 10：PyTorch 学習ループ 5 行の解剖図 -->

<!-- ========================================== -->

![スライド画像: PyTorch学習ループ5行の解剖](slides/10_pytorch_loop_anatomy.png)

> **【スライド 10 生成プロンプト】**

> - **用途**: PyTorch の標準コード 5 行と、その裏で動作する理論の1対1対応を示す解説スライド

> - **プロンプト（英語）**:

>   `Educational code explanation diagram on dark theme IDE interface. Shows 5 clean lines of PyTorch code: 1. optimizer.zero_grad(), 2. pred = model(X), 3. loss = criterion(pred, y), 4. loss.backward(), 5. optimizer.step(). Next to each code line, a glowing callout box explains the physical meaning (Reset memo, Forward inference, Evaluate loss, Reverse chain-rule gradients, Update W and b). Sleek presentation design, 16:9 ratio.`

> - **スライド記載テキスト**:

>   1. `zero_grad()`: 足元の傾きメモを白紙に戻す

>   2. `model(X)`: 現在の $W, b$ で順伝播を実行

>   3. `criterion(...)`: 正解とのズレ $L$ を算出

>   4. `loss.backward()`: 逆伝播で全 $W, b$ の傾きを一括計算

>   5. `optimizer.step()`: 傾きに従って $W, b$ を更新

---

### 4. エポック・イテレーション・バッチサイズの関係

用語の混乱を防ぐための整理です。

* **バッチサイズ**：1回のパラメータ更新に使うデータ数（例: 100件）

* **イテレーション**：パラメータを更新した**回数**（1回の更新＝1イテレーション）

* **エポック**：訓練データ全体を**ちょうど1周**すること

$$
\text{1エポックあたりのイテレーション数} = \left\lceil \frac{\text{全データ件数}}{\text{バッチサイズ}} \right\rceil
$$

```python
n_samples, batch_size, epochs = 1000, 100, 50
iters_per_epoch = n_samples // batch_size
print(f"1エポック = {iters_per_epoch} イテレーション")    # 10 回更新
print(f"合計更新回数 = {iters_per_epoch * epochs}")       # 500 回更新
```

---

## 3.4 最適化アルゴリズム（Optimizer）（5分）

### 1. なぜ「素朴な勾配降下法 (SGD)」だけでは不十分なのか？

「求めた傾きの逆へ $\eta$ だけ進む」という素朴な更新式（`optimizer.step()`）には、現実の複雑な地形において3つの弱点があります。

1. **細長い谷の罠**：谷底に向かいたいのに、両側の急な壁の間を行き来して**ジグザグに激しくバウンド**し、前に進まない

2. **平坦な場所（プラトー）の罠**：傾きがほぼ 0 の平坦な場所にくると、歩幅がゼロになって**完全に立ち往生**する

3. **歩幅の一律化**：すべてのパラメータに同じ歩幅 $\eta$ を使うため、急なパラメータは崖から吹き飛び、緩やかなパラメータは一生動かない

これらを解決するのが**最適化アルゴリズム（オプティマイザ）**です。

---

### 2. 解決のための2大アイデアと進化の系譜

主要なオプティマイザは、**2つのアイデアの組み合わせ**として理解すると一発で頭に入ります。

```
  アイデア①：慣性（Momentum）
    前回のスピードを引き継ぐ。ボールが坂を転がるように、
    ジグザグの横揺れを打ち消して前進を加速し、平坦な窪みも勢いで突破する。
  アイデア②：パラメータごとの歩幅自動調整（AdaGrad / RMSProp）
    よく動くパラメータは「危ないから歩幅を小さく」、
    あまり動かないパラメータは「遅れているから歩幅を大きく」自動調整する。
  ★ Adam ＝ アイデア①（慣性） ＋ アイデア②（歩幅自動調整）
```

| 手法 | 登場年 | 特徴 |
|---|:---:|---|
| **SGD** | — | 素朴な勾配降下。シンプルだがジグザグに弱く止まりやすい |
| **Momentum** | 1964 | 慣性を導入。ジグザグを抑えて収束を加速 |
| **AdaGrad** | 2011 | パラメータごとに歩幅を自動調整。ただし途中で歩幅が小さくなりすぎて止まりがち |
| **RMSProp** | 2012 | AdaGrad の欠点を移動平均で改善 |
| **Adam** | 2014 | Momentum ＋ RMSProp の合体。**現在のデファクト標準** |
| **AdamW** | 2017 | Adam の過学習抑制（Weight Decay）を正しく修正。**Transformer 系の標準** |

```python
import torch
params = model.parameters()
opt_sgd     = torch.optim.SGD(params, lr=0.01, momentum=0.9)
opt_rmsprop = torch.optim.RMSprop(params, lr=0.001)
opt_adam    = torch.optim.Adam(params, lr=0.001)       # ← 迷ったらまずこれ！
opt_adamw   = torch.optim.AdamW(params, lr=0.001)      # Transformer系ならこれ！
```

---

<!-- ========================================== -->

<!-- スライド画像 11：オプティマイザの進化マップ -->

<!-- ========================================== -->

![スライド画像: オプティマイザの進化系譜](slides/11_optimizer_evolution.png)

> **【スライド 11 生成プロンプト】**

> - **用途**: SGD から Momentum（慣性）と RMSProp（歩幅調整）を経て Adam / AdamW へと統合される進化マップ

> - **プロンプト（英語）**:

>   `Architecture evolution roadmap of deep learning optimizers. Starting from SGD on the left, splitting into two branches: Branch A "Momentum (Inertia)" and Branch B "RMSProp (Adaptive Learning Rate)". The branches merge into a central prominent hero card "Adam (Combined De-facto Standard)" and forward to "AdamW (Modern Standard for LLMs)". High tech diagram, dark background, glowing connection nodes, 16:9 ratio.`

> - **スライド記載テキスト**:

>   - 慣性（Momentum） ＋ 歩幅自動調整（RMSProp） ＝ **Adam**

>   - 現在の標準: 通常のタスクは `Adam`、LLM/Transformer は `AdamW`

---

<!-- ========================================== -->

<!-- スライド画像 12：谷底へのアプローチ比較（SGD vs Adam） -->

<!-- ========================================== -->

![スライド画像: SGD vs Adamの収束軌跡](slides/12_sgd_vs_adam_valley.png)

> **【スライド 12 生成プロンプト】**

> - **用途**: 狭い谷における素朴なSGDの激しいジグザグと、Adamの滑らかで直線的な収束軌跡の対比

> - **プロンプト（英語）**:

>   `Contour map of a narrow, steep ravine loss landscape showing optimization trajectories. Red dashed line: Standard SGD bouncing wildly back and forth between steep walls, making slow progress. Green glowing solid line: Adam optimizer smoothly dampening oscillations and accelerating directly down the ravine center toward the minimum point. Scientific visualization style, dark mode, 16:9 ratio.`

> - **スライド記載テキスト**:

>   - 赤（SGD）: 「壁の間を激しくジグザグして前に進めない」

>   - 緑（Adam）: 「横揺れを打ち消し、歩幅を最適化して最短で谷底へ直進」

---

### 3. 実務での選択指針

1. **迷ったらまず **`**Adam(lr=1e-3)**`** で始める。** 大半のタスクで安定して動作します。

2. **Transformer や LLM を扱うなら **`**AdamW**`**。**

3. **画像分類で最後の数%の極限精度を絞り出したいなら **`**SGD + Momentum**`**。** ただし学習率のスケジュール調整が必要で手間がかかります。

> **よくある誤解**  

> 「オプティマイザを変えれば精度が劇的に上がる」と期待されがちですが、効果は限定的です。**データの質・量、ネットワーク構造、学習率**の方がはるかに支配的です。Adam で動かないモデルが、別のオプティマイザに変えただけで急に動くことはほぼありません。

---

## 本章のまとめ（2分）

✅ **前提**: 動かせるツマミは $W$（重要度）と $\mathbf{b}$（しきい値）のみ。初期値はランダム。  

✅ **損失関数**: 「今のズレ」を微分可能な数値にする。回帰は MSE、分類は交差エントロピー。  

✅ **勾配降下法**: 「足元の傾きと逆向きに一歩進む」基本ルール。歩幅は学習率 $\eta$ で決まる。  

✅ **誤差逆伝播法**: 連鎖律の応用。1回の逆向き走査で全パラメータの傾きを一括計算する。  

✅ **PyTorch 5行の型**: `zero_grad` $\rightarrow$ `forward` $\rightarrow$ `loss` $\rightarrow$ `backward` $\rightarrow$ `step`。  

✅ **オプティマイザ**: 慣性と歩幅調整を両立した **Adam / AdamW** が実務の標準。

---

## 第4章へのブリッジ（受講生の興味を引く締めくくり）

> 「本章の仕組みによって、モデルはデータをもとに $W$ と $\mathbf{b}$ を自動調整し、損失 $L$ をゼロ近くまで減らせるようになりました。

> 

> しかし、ここで1つの重大な問題が発生します。

> **『訓練データの損失がゼロになった＝本当に賢いモデルが完成した』と言えるでしょうか？**

> 

> 過去問の答えを丸暗記して満点を取っても、本番の初見問題が解けなければ意味がありません。

> 訓練データを暗記してしまう**過学習（Overfitting）**をどう防ぎ、モデルの真の賢さをどう測るのか。

> それを扱うのが次の第4章です！」

---

## 演習問題

1. 3クラス分類で、あるサンプルの正解がクラス1だったとします。次の2つの予測のうち、交差エントロピー損失が大きいのはどちらですか。実際に $-\log(p)$ を計算して確かめてください。

   - 予測A: `[0.3, 0.4, 0.3]`

   - 予測B: `[0.8, 0.1, 0.1]`

2. 3.2 の勾配降下のコードで、学習率 `lr` を `0.01` / `0.5` / `1.1` に変えて実行し、それぞれ何が起きるか観察してください。`lr = 1.0` にしたときの挙動も予想してから試してみましょう。

3. 訓練データが 12,800 件、バッチサイズ 64、エポック数 30 で学習するとき、パラメータの更新（イテレーション）は合計何回行われますか。

4. 次の学習ループには 1 箇所致命的なバグがあります。指摘して修正してください。また、そのまま実行すると学習にどんな異常が現れるか説明してください。

```python
for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```
