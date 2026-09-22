import pandas as pd
import os

# 读取标签文件
labels = pd.read_csv('../train_labels.csv', encoding='utf-8-sig')
print('标签表前5行：')
print(labels.head())
print('\n标签数量统计：')
print(labels['label'].value_counts())

# 检查 train 文件夹
train_files = os.listdir('../train')
print('\ntrain 文件夹文件数量：', len(train_files))
print('前5个文件：', train_files[:5])

# 检查 test 文件夹
test_files = os.listdir('../test')
print('\ntest 文件夹文件数量：', len(test_files))
print('test 文件列表：', test_files)

# 检查标签里的编号是否都能在 train 文件夹找到对应文件
missing = []
for sid in labels['subject_id']:
    fname = sid + '_T1w.nii.gz'
    if fname not in train_files:
        missing.append(fname)
print('\n标签中缺失对应训练文件的数量：', len(missing))
if missing:
    print('缺失示例：', missing[:5])