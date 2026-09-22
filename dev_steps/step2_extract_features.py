import os
import warnings
import pandas as pd
import numpy as np
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img, load_img
from nilearn.maskers import NiftiLabelsMasker

# 忽略所有警告，让输出变得干净
warnings.filterwarnings('ignore')

print("正在准备标准脑模板...")
mni = load_mni152_template(resolution=2)

print("正在加载本地的 AAL3 脑图谱...")
# 这里指向你本地解压出来的文件路径
aal_path = '../AAL3/AAL3v1.nii'

masker = NiftiLabelsMasker(
    labels_img=aal_path,
    strategy="mean",
    standardize=None          # 把 False 改成 None，消除 FutureWarning
)

# 读取标签和训练集编号
labels = pd.read_csv('../train_labels.csv', encoding='utf-8-sig')
train_ids = labels['subject_id'].tolist()
y = labels['label'].values

# 读取测试集文件列表
test_files = sorted(os.listdir('../test'))
test_ids = [f.replace('_T1w.nii.gz', '') for f in test_files]

# 提取训练集特征
print("开始提取训练集特征，共有", len(train_ids), "个受试者...")
train_features = []
for i, sid in enumerate(train_ids):
    path = f'train/{sid}_T1w.nii.gz'
    img = load_img(path)
    # 将原始图粗略重采样到标准脑空间
    img_resampled = resample_to_img(img, mni, interpolation='continuous')
    # 按脑区提取数值
    features = masker.fit_transform(img_resampled)
    train_features.append(features.flatten())
    if (i+1) % 20 == 0:
        print(f"已完成 {i+1}/{len(train_ids)}")

# 提取测试集特征
print("开始提取测试集特征，共有", len(test_files), "个受试者...")
test_features = []
for i, f in enumerate(test_files):
    path = f'test/{f}'
    img = load_img(path)
    img_resampled = resample_to_img(img, mni, interpolation='continuous')
    features = masker.transform(img_resampled)
    test_features.append(features.flatten())

# 保存提取好的特征表
X_train = pd.DataFrame(train_features)
X_train['label'] = y
X_train.to_csv('X_train.csv', index=False)

X_test = pd.DataFrame(test_features)
X_test['subject_id'] = test_ids
X_test.to_csv('X_test.csv', index=False)

print("特征提取完成！")
print("训练集特征形状（行数，列数）：", X_train.shape)
print("测试集特征形状（行数，列数）：", X_test.shape)