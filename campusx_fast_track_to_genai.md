# CampusX Fast Track to GenAI — ML + DL Watchlist

Goal: quickly finish the useful parts of CampusX ML and DL, then move to GenAI/LangChain.  
Rule: **watch only this file first; skip everything else for now.**

Legend:
- ✅ = must watch for your route toward GenAI
- 🧠 = concept/theory; no dataset needed
- 📦 = dataset/code available
- ⚠️ = I could not confirm an official public dataset for that specific video; use the video/code notes only

---

## 1) Machine Learning Fast Route

### A. ML orientation — watch fast, 1.25x/1.5x

| Done | Day | Video | Why watch | Dataset / code |
|---|---:|---|---|---|
| [ ] | 1 | [What is Machine Learning?](https://www.youtube.com/watch?v=ZftI2fEz0Fw) | Basic ML idea | 🧠 No dataset needed |
| [ ] | 2 | [AI Vs ML Vs DL for Beginners](https://www.youtube.com/watch?v=1v3_AQ26jZ0) | Understand AI/ML/DL difference before GenAI | 🧠 No dataset needed |
| [ ] | 3 | [Types of Machine Learning](https://www.youtube.com/watch?v=81ymPYEtFOw) | Supervised/unsupervised/reinforcement overview | 🧠 No dataset needed |
| [ ] | 7 | [Challenges in Machine Learning](https://www.youtube.com/watch?v=WGUNAJki2S4) | Know real ML problems | 🧠 No dataset needed |
| [ ] | 9 | [Machine Learning Development Life Cycle](https://www.youtube.com/watch?v=iDbhQGz_rEo) | Project workflow thinking | 🧠 No dataset needed |
| [ ] | 11 | [What are Tensors](https://www.youtube.com/watch?v=vVhD2EyS41Y) | Helps before DL/transformers | 🧠 No dataset needed |
| [ ] | 13 | [End to End Toy Project](https://www.youtube.com/watch?v=dr7z7a_8lQw) | See full ML workflow once | ⚠️ Use video/code notes; official direct dataset not confirmed |
| [ ] | 14 | [How to Frame a Machine Learning Problem](https://www.youtube.com/watch?v=A9SezQlvakw) | Very useful for AI product thinking | 🧠 No dataset needed |

### B. Data + preprocessing — important for real AI/RAG apps

| Done | Day | Video | Why watch | Dataset / code |
|---|---:|---|---|---|
| [ ] | 19 | [Understanding Your Data](https://www.youtube.com/watch?v=mJlRTUuVr04) | Core EDA/data reading | 📦 [train.csv](https://github.com/campusx-official/100-days-of-machine-learning/raw/main/day19-understanding-your-data-descriptive-stats/train.csv) / [GitHub folder](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day19-understanding-your-data-descriptive-stats) |
| [ ] | 20 | [EDA using Univariate Analysis](https://www.youtube.com/watch?v=4HyTlbHUKSw) | Understand one-column analysis | ⚠️ Official direct dataset not confirmed; likely same Titanic-style EDA data |
| [ ] | 21 | [EDA using Bivariate and Multivariate Analysis](https://www.youtube.com/watch?v=6D3VtEfCw7w) | Understand relationships between features | ⚠️ Official direct dataset not confirmed |
| [ ] | 23 | [What is Feature Engineering](https://www.youtube.com/watch?v=sluoVhT0ehg) | Feature thinking matters in ML products | 🧠 No dataset needed |
| [ ] | 24 | [Feature Scaling - Standardization](https://www.youtube.com/watch?v=1Yw9sC0PNwY) | Needed for many ML models | 📦 [Social_Network_Ads.csv](https://github.com/campusx-official/100-days-of-machine-learning/raw/main/day24-standardization/Social_Network_Ads.csv) / [GitHub folder](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day24-standardization) |
| [ ] | 25 | [Feature Scaling - Normalization](https://www.youtube.com/watch?v=eBrGyuA2MIg) | Know standardization vs normalization | ⚠️ Official direct dataset not confirmed |
| [ ] | 26 | [Encoding Categorical Data](https://www.youtube.com/watch?v=w2GglmYHfmM) | Crucial preprocessing | 📦 [customer.csv](https://github.com/campusx-official/100-days-of-machine-learning/raw/main/day26-ordinal-encoding/customer.csv) / [GitHub folder](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day26-ordinal-encoding) |
| [ ] | 27 | [One Hot Encoding](https://www.youtube.com/watch?v=U5oCv3JKWKA) | Crucial preprocessing | ⚠️ Official direct dataset not confirmed |
| [ ] | 28 | [Column Transformer](https://www.youtube.com/watch?v=5TVj6iEBR4I) | Important for clean sklearn pipelines | ⚠️ Official direct dataset not confirmed / [GitHub folder](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day28-column-transformer) |
| [ ] | 29 | [Machine Learning Pipelines A-Z](https://www.youtube.com/watch?v=xOccYkgRV4Q) | Very important for production workflow | 📦 [train.csv](https://github.com/campusx-official/100-days-of-machine-learning/raw/main/day29-sklearn-pipelines/train.csv) / [GitHub folder](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day29-sklearn-pipelines) |
| [ ] | 35 | [Handling Missing Data: Complete Case Analysis](https://www.youtube.com/watch?v=aUnNWZorGmk) | Missing data basics | ⚠️ Official direct dataset not confirmed |
| [ ] | 36 | [Handling Missing Numerical Data: Simple Imputer](https://www.youtube.com/watch?v=mCL2xLBDw8M) | Missing numeric values | ⚠️ Official direct dataset not confirmed |
| [ ] | 37 | [Handling Missing Categorical Data](https://www.youtube.com/watch?v=l_Wip8bEDFQ) | Missing categorical values | ⚠️ Official direct dataset not confirmed |
| [ ] | 41 | [What are Outliers](https://www.youtube.com/watch?v=Lln1PKgGr_M) | Basic outlier concept | 🧠 No dataset needed |
| [ ] | 42 | [Outlier Detection using Z-score](https://www.youtube.com/watch?v=OnPE-Z8jtqM) | Common outlier method | ⚠️ Official direct dataset not confirmed |

### C. Core ML algorithms — only the minimum before DL/GenAI

| Done | Day | Video | Why watch | Dataset / code |
|---|---:|---|---|---|
| [ ] | 48/50 | [Simple Linear Regression](https://www.youtube.com/watch?v=UZPfbG0jNec) | Basic regression intuition | ⚠️ Official direct dataset not confirmed |
| [ ] | 49/52 | [Regression Metrics: MSE, MAE, RMSE, R2](https://www.youtube.com/watch?v=Ti7c-Hz7GSM) | Know how models are judged | 🧠 No dataset needed |
| [ ] | 50 | [Multiple Linear Regression](https://www.youtube.com/watch?v=ashGekqstl8) | Practical regression with multiple features | 📦 [GitHub folder](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day50-multiple-linear-regression) |
| [ ] | 51/56 | [Gradient Descent](https://www.youtube.com/watch?v=7z6yXpYk7sw) | Needed before neural networks | 🧠 No dataset needed |
| [ ] | 52/59 | [Mini-Batch Gradient Descent](https://www.youtube.com/watch?v=_scscQ4HVTY) | Needed before DL optimizers | 🧠 No dataset needed |
| [ ] | 61 | [Bias Variance Trade-off](https://www.youtube.com/watch?v=74DU02Fyrhk) | Understand overfitting | 🧠 No dataset needed |
| [ ] | 69 | [Logistic Regression Part 1](https://www.youtube.com/watch?v=XNXzVfItWGY) | Classification basics | 🧠 No dataset needed |
| [ ] | 75 | [Accuracy and Confusion Matrix](https://www.youtube.com/watch?v=c09drtuCS3c) | Classification metrics | 🧠 No dataset needed |
| [ ] | 76 | [Precision, Recall and F1 Score](https://www.youtube.com/watch?v=iK-kdhJ-7yI) | Very important for AI/ML evaluation | 🧠 No dataset needed |
| [ ] | 80 | [Decision Trees: Entropy, Gini, Information Gain](https://www.youtube.com/watch?v=IZnno-dKgVQ) | Tree models are used inside ensembles | 🧠 No dataset needed |
| [ ] | 84 | [Introduction to Ensemble Learning](https://www.youtube.com/watch?v=bHK1fE_BUms) | Understand why ensembles work | 🧠 No dataset needed |
| [ ] | 91 | [Introduction to Random Forest](https://www.youtube.com/watch?v=F9uESCHGjhA) | Strong classical ML model | 📦 [GitHub folder](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day65-random-forest) |
| [ ] | 103 | [K-Means Clustering Algorithm](https://www.youtube.com/watch?v=5shTLzwAdEc) | Basic unsupervised learning | 📦 [GitHub folder](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/kmeans) |
| [ ] | 127 | [Introduction to XGBoost / XGBoost Regression](https://www.youtube.com/watch?v=gmp2tS2joaA) | Know what XGBoost is; do not go deep now | 🧠 No dataset needed for first pass |
| [ ] | 132 | [Imbalanced Data: Under/Oversampling/SMOTE](https://www.youtube.com/watch?v=yh2AKoJCV3k) | Important for real classification | ⚠️ Dataset not confirmed |
| [ ] | 134 | [ROC-AUC / Classification Curve Concept](https://www.youtube.com/watch?v=uKPAeT5Z5QE) | Important evaluation concept | 🧠 No dataset needed |

---

## 2) Deep Learning Fast Route

### A. ANN foundation

| Done | Day | Video | Why watch | Dataset / code |
|---|---:|---|---|---|
| [ ] | 2 | [What is Deep Learning? Deep Learning Vs Machine Learning](https://www.youtube.com/watch?v=fHF22Wxuyw4) | DL overview | 🧠 No dataset needed |
| [ ] | 3 | [Types of Neural Networks / History / Applications](https://www.youtube.com/watch?v=fne_UE7hDn0) | Know ANN/CNN/RNN/transformer families | 📦 [Official DL GitHub day3](https://github.com/campusx-official/100-days-of-deep-learning/tree/main/day3) |
| [ ] | 4 | [What is a Perceptron?](https://www.youtube.com/watch?v=X7iIKPoZ0Sw) | Neural network basic unit | 📦 [Official DL GitHub day4](https://github.com/campusx-official/100-days-of-deep-learning/tree/main/day4) |
| [ ] | 7 | [Problem with Perceptron](https://www.youtube.com/watch?v=Jp44b27VnOg) | Why MLP is needed | 🧠 No dataset needed |
| [ ] | 8 | [MLP Notation](https://www.youtube.com/watch?v=H0_3SJh4Rqs) | Understand layer notation | 🧠 No dataset needed |
| [ ] | 9 | [Multi Layer Perceptron / MLP Intuition](https://www.youtube.com/watch?v=qw7wFGgNCSU) | Core ANN intuition | 🧠 No dataset needed |
| [ ] | 10 | [Forward Propagation](https://www.youtube.com/watch?v=7MuiScUkboE) | Understand how NN predicts | 🧠 No dataset needed |
| [ ] | 11 | [Customer Churn Prediction using ANN](https://www.youtube.com/watch?v=9wmImImmgcI) | Practical ANN classification | ⚠️ Official direct dataset not confirmed |
| [ ] | 12 | [Handwritten Digit Classification using ANN / MNIST](https://www.youtube.com/watch?v=3xPT2Pk0Jds) | Practical ANN on MNIST | 📦 Dataset: built-in MNIST via Keras/TensorFlow |

### B. Training fundamentals

| Done | Day | Video | Why watch | Dataset / code |
|---|---:|---|---|---|
| [ ] | 14 | [Loss Functions in Deep Learning](https://www.youtube.com/watch?v=gb5nm_3jBIo) | Understand model loss | 🧠 No dataset needed |
| [ ] | 15 | [Backpropagation Part 1: The What](https://www.youtube.com/watch?v=6M1wWQmcUjQ) | Most important DL training concept | 🧠 No dataset needed |
| [ ] | 16 | [Backpropagation Part 2: The How](https://www.youtube.com/watch?v=ma6hWrU-LaI) | How gradients train networks | 🧠 No dataset needed |
| [ ] | 19 | [Gradient Descent in Neural Networks](https://www.youtube.com/watch?v=7z6yXpYk7sw) | DL optimization basics | 🧠 No dataset needed |
| [ ] | 20 | [Vanishing / Exploding Gradient Problem](https://www.youtube.com/watch?v=uCrevbBh0zM) | Needed before RNN/LSTM/transformers | 🧠 No dataset needed |
| [ ] | 22 | [Early Stopping](https://www.youtube.com/watch?v=Ygvskt5HadI) | Prevent overfitting | 🧠 No dataset needed |
| [ ] | 23 | [Data Scaling in Neural Network](https://www.youtube.com/watch?v=mzRO0cVppQ0) | Important for stable training | 🧠 No dataset needed |
| [ ] | 24 | [Dropout Layer](https://www.youtube.com/watch?v=gyTlcHVeBjM) | Regularization technique | 🧠 No dataset needed |
| [ ] | 26 | [Regularization in Deep Learning](https://www.youtube.com/watch?v=4xRonrhtkzc) | L1/L2/weight decay basics | 🧠 No dataset needed |
| [ ] | 27 | [Activation Functions](https://www.youtube.com/watch?v=7LcUkgzx3AY) | ReLU/sigmoid/tanh basics | 🧠 No dataset needed |
| [ ] | 29 | [Weight Initialization Techniques](https://www.youtube.com/watch?v=2MSY0HwH5Ss) | Avoid bad training | 🧠 No dataset needed |
| [ ] | 30 | [Xavier/Glorot and He Initialization](https://www.youtube.com/watch?v=nwVOSgcrbQI) | Practical weight init | 🧠 No dataset needed |
| [ ] | 31 | [Batch Normalization](https://www.youtube.com/watch?v=2AscwXePInA) | Very useful in DL | 🧠 No dataset needed |
| [ ] | 32 | [Optimizers in Deep Learning Part 1](https://www.youtube.com/watch?v=iCTTnQJn50E) | Optimizer basics | 🧠 No dataset needed |

### C. RNN → Attention → Transformer path — this is the key GenAI bridge

| Done | Day | Video | Why watch | Dataset / code |
|---|---:|---|---|---|
| [ ] | 55 | [Why RNNs are Needed](https://www.youtube.com/watch?v=4KpRP-YUw6c) | Understand sequence models | 🧠 No dataset needed |
| [ ] | 56 | [RNN Forward Propagation / Architecture](https://www.youtube.com/watch?v=BjWqCcbusMM) | Basic RNN architecture | 🧠 No dataset needed |
| [ ] | 61 | [LSTM Part 1: What and Why](https://www.youtube.com/watch?v=z7IPBg6MyrU) | Understand why LSTM existed before transformers | 🧠 No dataset needed |
| [ ] | 67 | [History of LLMs](https://www.youtube.com/watch?v=8fX3rOjTloc) | Direct bridge to GenAI | 🧠 No dataset needed |
| [ ] | 68 | [Encoder Decoder / Sequence-to-Sequence](https://www.youtube.com/watch?v=KiL74WsgxoA) | Needed before attention/transformers | 🧠 No dataset needed |
| [ ] | 69 | [Attention Mechanism in 1 Video](https://www.youtube.com/watch?v=rj5V6q6-XUM) | Extremely important for transformers | 🧠 No dataset needed |
| [ ] | 71 | [Introduction to Transformers](https://www.youtube.com/watch?v=BjRVS2wTtcA) | Must-watch before GenAI | 🧠 No dataset needed |
| [ ] | 72 | [What is Self Attention](https://www.youtube.com/watch?v=XnGGmvpDLA0) | Core transformer mechanism | 🧠 No dataset needed |
| [ ] | 73 | [Self Attention with Code](https://www.youtube.com/watch?v=-tCKPl_8Xb8) | See self-attention practically | ⚠️ Code/dataset not confirmed |
| [ ] | 74 | [Scaled Dot Product Attention](https://www.youtube.com/watch?v=r7mAt0iVqwo) | Transformer attention math | 🧠 No dataset needed |
| [ ] | 77 | [Multi-Head Attention](https://www.youtube.com/watch?v=bX2QwpjsmuA) | Core LLM architecture piece | 🧠 No dataset needed |
| [ ] | 81 | [Masked Self Attention](https://www.youtube.com/watch?v=m6onaKFzF94) | Important for decoder-only LLMs | 🧠 No dataset needed |
| [ ] | 82 | [Cross Attention](https://www.youtube.com/watch?v=smOnJtCevoU) | Important for encoder-decoder models | 🧠 No dataset needed |
| [ ] | 83 | [Transformer Decoder Architecture](https://www.youtube.com/watch?v=DI2_hrAulYo) | Very important for GPT-style models | 🧠 No dataset needed |

---

## Final instruction

After completing this file, start CampusX GenAI/LangChain immediately. Do **not** wait to finish every skipped ML/DL lecture.

For GenAI, your real bridge is:

**Python → data cleaning → model evaluation → neural networks → backprop/optimizers → sequence models → attention → transformers → LangChain/RAG/agents.**

---

## Official source links

- ML playlist/course preview: https://learnwith.campusx.in/s/preview/courses/100-Days-of-Machine-Learning-YouTube-1763210027804-6918732bee601b12112865e6
- ML GitHub repo: https://github.com/campusx-official/100-days-of-machine-learning
- DL playlist/course preview: https://learnwith.campusx.in/s/preview/courses/100-Days-of-Deep-Learning-YouTube-6818549cc1a70a79d992169f
- DL GitHub repo: https://github.com/campusx-official/100-days-of-deep-learning
