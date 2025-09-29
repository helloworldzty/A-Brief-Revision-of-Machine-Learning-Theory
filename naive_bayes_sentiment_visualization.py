import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
# MultinomialNB：多项式朴素贝叶斯模型，特别适用于离散特征（如单词计数）的分类，是文本分类的经典算法。
# CountVectorizer：文本特征提取器。它将文本数据转换为模型可以理解的数值格式（词频矩阵）。
# PCA：主成分分析，一种降维技术，用于将高维数据投影到二维平面进行可视化。

positive_words = ["good", "excellent", "fantastic", "amazing", "wonderful", "love", "great", "awesome", "nice", "pleasant"]
negative_words = ["bad", "terrible", "awful", "horrible", "hate", "worst", "boring", "poor", "dull", "disappointing"]

np.random.seed(0);
texts = []
labels = []

for _ in range(20): # 循环20次，生成20个正面样本
    sample = " ".join(np.random.choice(positive_words, size=3, replace=True)) # 从正面词汇中随机选择3个词（可重复），用空格连接成一个句子
    texts.append(sample) # 将句子加入texts列表
    labels.append(1) # 将标签1（正面）加入labels列表
for _ in range(20): # 循环20次，生成20个负面样本
    sample = " ".join(np.random.choice(negative_words, size=3, replace=True))
    texts.append(sample)
    labels.append(0)

#文本向量化（特征提取）
vectorizer = CountVectorizer() # 初始化一个CountVectorizer对象
X = vectorizer.fit_transform(texts) # 拟合（学习词汇表）并转换（将文本转换为词频矩阵）训练数据（特征矩阵）
# 这是机器学习中至关重要的一步——特征工程
# CountVectorizer将文本集合转换为一个词频矩阵
# fit_transform方法做两件事：
# fit：学习数据中的词汇表（所有唯一的单词）。
# transform：根据学到的词汇表，将每条文本转换成一个向量。向量的每个元素代表一个特定单词在该文本中出现的次数。
# 例如，词汇表是 ['amazing', 'bad', 'good', ...]，句子 "good good bad"会被转换为向量 [0, 1, 2, ...]。

nb = MultinomialNB() # 初始化一个多项式朴素贝叶斯分类器
nb.fit(X, labels) # 使用训练数据（特征矩阵X和标签labels）来训练模型

test_texts = [ # 定义测试句子列表
    "good excellent fantastic",   # 纯正面 -> 预期预测为1
    "bad terrible awful",         # 纯负面 -> 预期预测为0
    "amazing wonderful love",     # 纯正面 -> 预期预测为1
    "boring dull disappointing",  # 纯负面 -> 预期预测为0
    "awesome awesome bad",        # 混合（2正1负）-> 模型会进行“概率投票”
    "worst pleasant horrible",    # 混合（1正2负）-> 模型会进行“概率投票”
]
X_test = vectorizer.transform(test_texts) # 使用训练时学习的词汇表来转换测试文本
y_pred = nb.predict(X_test) # 使用训练好的模型对转换后的测试数据进行预测

pca = PCA(n_components=2, random_state=42) # 初始化PCA，将数据降维到2个主成分
X_all = np.vstack([X.toarray(), X_test.toarray()]) # 将训练和测试特征矩阵在垂直方向堆叠
X_all_2d = pca.fit_transform(X_all) # 对合并后的数据进行拟合和降维
X_2d = X_all_2d[:len(X.toarray())] # 拆分出训练集在2维空间中的坐标
X_test_2d = X_all_2d[len(X.toarray()):] # 拆分出测试集在2维空间中的坐标

plt.figure(figsize=(8, 6)) # 创建一个新的图形窗口
colors = ['red' if label == 1 else 'blue' for label in labels] # 根据标签生成颜色列表：正面为红色，负面为蓝色
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6, label='Training data') # 绘制训练数据的散点图
test_colors = ['green' if pred == 1 else 'purple' for pred in y_pred] # 根据预测结果生成颜色列表：正面为绿色，负面为紫色
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=test_colors, marker='*', s=200, edgecolor='k', label='Test data') # 用星号绘制测试数据点
for i, txt in enumerate(test_texts):
    plt.annotate(f"{txt}\n(pred={y_pred[i]})", (X_test_2d[i, 0]+0.2, X_test_2d[i, 1]), fontsize=8, color=test_colors[i]) # 为每个测试点添加文本标注
plt.xlabel('PCA Component 1') # X轴标签
plt.ylabel('PCA Component 2') # Y轴标签
plt.title('Naive Bayes Text Classification Visualization (Synthetic Data)') # 图表标题
plt.legend() # 显示图例
plt.grid(True) # 显示网格
plt.tight_layout() # 自动调整布局
plt.show() # 显示图形

for text, pred in zip(test_texts, y_pred): # 遍历测试文本和预测结果
    print(f'"{text}": predicted class {pred}') # 格式化输出