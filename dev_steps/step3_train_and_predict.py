import warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

# 忽略警告
warnings.filterwarnings('ignore')

print("正在读取特征表...")
train_data = pd.read_csv('../X_train.csv')
test_data = pd.read_csv('../X_test.csv')

X_train = train_data.drop('label', axis=1)
y_train = train_data['label'].values
test_ids = test_data['subject_id'].values
X_test = test_data.drop('subject_id', axis=1)

print("训练集形状：", X_train.shape)
print("测试集形状：", X_test.shape)

# 准备跑三次
for round_num in range(1, 4):
    print(f"\n========== 第 {round_num} 次预测 ==========")

    # 每次使用不同的随机种子，让数据划分和模型训练略有差异
    rs = round_num * 10

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=rs))
    ])

    # 5折交叉验证评估（用于报告里写模拟考成绩）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)

    acc = accuracy_score(y_train, y_pred_cv)
    cm = confusion_matrix(y_train, y_pred_cv, labels=[1, 2])
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print(f"模拟考成绩 -> 总正确率: {acc:.4f}，自闭症识别率: {sensitivity:.4f}，健康人识别率: {specificity:.4f}")

    # 用全部136人训练最终模型
    model.fit(X_train, y_train)

    # 预测10个未知样本
    probs = model.predict_proba(X_test)
    asd_probs = probs[:, 0]  # 自闭症概率

    # 题目已知：10人里是5个ASD、5个HC
    # 按自闭症概率从高到低排序，前5个填1，后5个填2
    sorted_indices = np.argsort(asd_probs)[::-1]
    final_labels = np.ones(10, dtype=int)
    final_labels[sorted_indices[5:]] = 2

    # 生成提交文件
    submission = pd.DataFrame({'subject_id': test_ids, 'label': final_labels})
    submission = submission.set_index('subject_id').loc[test_ids].reset_index()

    # 保存为不同的文件名
    filename = f'submission_{round_num}.csv'
    submission.to_csv(filename, index=False)

    print(f"第 {round_num} 次预测结果：")
    print(submission.to_string(index=False))
    print(f"已保存为 {filename}")

print("\n===== 三次预测全部结束 =====")
print("请在左侧项目文件夹找到 submission_1.csv、submission_2.csv、submission_3.csv，全部提交上去。")