# -*- coding: utf-8 -*-
"""
ASD/HC 脑影像分类系统
功能：读取T1加权脑影像，提取AAL3脑区特征，用三种不同模型分类，输出三次预测结果。
作者：XXX
日期：2026-XX-XX
"""

import os
import queue
import threading
import warnings
import pickle

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext, messagebox

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
TRAIN_DIR = 'train'
TEST_DIR = 'test'
LABEL_FILE = 'train_labels.csv'
AAL_PATH = 'AAL3/AAL3v1.nii'
X_TRAIN_FILE = 'X_train.csv'
X_TEST_FILE = 'X_test.csv'
MODEL_FILES = ['model_1.pkl', 'model_2.pkl', 'model_3.pkl']
SUBMISSION_FILES = ['submission_1.csv', 'submission_2.csv', 'submission_3.csv']
N_ROUNDS = 3


# ==================== 主界面 ====================
class ASDHCApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASD/HC 脑影像分类系统")
        self.root.geometry("1020x750")

        self.msg_queue = queue.Queue()
        self.busy = False

        self.build_ui()
        self.poll_queue()

    def build_ui(self):
        # 顶部按钮栏
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=12)

        self.btn_check = tk.Button(btn_frame, text="1. 检查数据",
                                   width=14, height=2, command=self.on_check)
        self.btn_check.pack(side=tk.LEFT, padx=6)

        self.btn_extract = tk.Button(btn_frame, text="2. 提取特征",
                                     width=14, height=2, command=self.on_extract)
        self.btn_extract.pack(side=tk.LEFT, padx=6)

        self.btn_train = tk.Button(btn_frame, text="3. 训练模型",
                                   width=14, height=2, command=self.on_train)
        self.btn_train.pack(side=tk.LEFT, padx=6)

        self.btn_predict = tk.Button(btn_frame, text="4. 预测测试集",
                                     width=14, height=2, command=self.on_predict)
        self.btn_predict.pack(side=tk.LEFT, padx=6)

        # 日志区
        tk.Label(self.root, text="进度日志：", anchor='w',
                 font=("微软雅黑", 10, "bold")).pack(fill=tk.X, padx=12)
        self.log_text = scrolledtext.ScrolledText(
            self.root, height=16, font=("Consolas", 10), bg="#f5f5f5")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # 底部区域（分为左右两部分）
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=12, pady=4)

        # 左下：评估指标
        left_frame = tk.Frame(bottom_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tk.Label(left_frame, text="评估指标：", anchor='w',
                 font=("微软雅黑", 10, "bold")).pack(fill=tk.X)
        self.metric_text = tk.Text(left_frame, height=8,
                                   font=("Consolas", 10), bg="#fffbe6")
        self.metric_text.pack(fill=tk.BOTH, expand=True)

        # 右下：预测结果
        right_frame = tk.Frame(bottom_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        tk.Label(right_frame, text="预测结果：", anchor='w',
                 font=("微软雅黑", 10, "bold")).pack(fill=tk.X)
        self.pred_text = scrolledtext.ScrolledText(right_frame, height=8,
                                                   font=("Consolas", 10), bg="#e6f7ff")
        self.pred_text.pack(fill=tk.BOTH, expand=True)

    # -------- 线程与消息队列 --------
    def log(self, msg):
        self.msg_queue.put(('log', msg))

    def set_metric(self, msg):
        self.msg_queue.put(('metric', msg))

    def clear_metric(self):
        self.msg_queue.put(('clear_metric', ''))

    def set_prediction(self, msg):
        self.msg_queue.put(('prediction', msg))

    def clear_prediction(self):
        self.msg_queue.put(('clear_prediction', ''))

    def poll_queue(self):
        try:
            while True:
                kind, msg = self.msg_queue.get_nowait()
                if kind == 'log':
                    self.log_text.insert(tk.END, msg + "\n")
                    self.log_text.see(tk.END)
                elif kind == 'metric':
                    self.metric_text.insert(tk.END, msg + "\n")
                    self.metric_text.see(tk.END)
                elif kind == 'clear_metric':
                    self.metric_text.delete("1.0", tk.END)
                elif kind == 'prediction':
                    self.pred_text.insert(tk.END, msg + "\n")
                    self.pred_text.see(tk.END)
                elif kind == 'clear_prediction':
                    self.pred_text.delete("1.0", tk.END)
        except queue.Empty:
            pass
        self.root.after(120, self.poll_queue)

    def run_in_thread(self, target):
        if self.busy:
            messagebox.showwarning("提示", "当前有任务正在运行，请稍候。")
            return
        self.busy = True

        def wrapper():
            try:
                target()
            except Exception as e:
                self.log(f"[错误] {type(e).__name__}: {e}")
            finally:
                self.busy = False
                self.log("")

        threading.Thread(target=wrapper, daemon=True).start()

    # -------- 四个按钮的入口 --------
    def on_check(self):
        self.run_in_thread(self.task_check)

    def on_extract(self):
        self.run_in_thread(self.task_extract)

    def on_train(self):
        self.run_in_thread(self.task_train)

    def on_predict(self):
        self.run_in_thread(self.task_predict)

    # ==================== 任务1：检查数据 ====================
    def task_check(self):
        self.log("========== 开始检查数据 ==========")

        if not os.path.exists(LABEL_FILE):
            self.log(f"[错误] 找不到标签文件：{LABEL_FILE}")
            return
        if not os.path.isdir(TRAIN_DIR):
            self.log(f"[错误] 找不到训练集文件夹：{TRAIN_DIR}")
            return
        if not os.path.isdir(TEST_DIR):
            self.log(f"[错误] 找不到测试集文件夹：{TEST_DIR}")
            return

        labels = pd.read_csv(LABEL_FILE, encoding='utf-8-sig')
        n_asd = int((labels['label'] == 1).sum())
        n_hc = int((labels['label'] == 2).sum())
        self.log(f"训练集样本总数：{len(labels)}")
        self.log(f"  其中 ASD（标签1）：{n_asd} 例")
        self.log(f"  其中 HC （标签2）：{n_hc} 例")

        train_files = os.listdir(TRAIN_DIR)
        test_files = os.listdir(TEST_DIR)
        self.log(f"train 文件夹文件数：{len(train_files)}")
        self.log(f"test  文件夹文件数：{len(test_files)}")

        missing = [sid for sid in labels['subject_id']
                   if f"{sid}_T1w.nii.gz" not in train_files]
        if missing:
            self.log(f"[警告] 有 {len(missing)} 个标签在训练文件夹里找不到对应文件。")
        else:
            self.log("训练集标签与文件一一对应，检查通过。")

        if os.path.exists(AAL_PATH):
            self.log(f"AAL3 图谱文件已就位：{AAL_PATH}")
        else:
            self.log(f"[警告] 未找到 AAL3 图谱：{AAL_PATH}")
            self.log("        请确认 AAL3 文件夹已放在项目根目录下。")

        self.log("========== 数据检查完毕 ==========")

    # ==================== 任务2：提取特征 ====================
    def task_extract(self):
        self.log("========== 开始提取特征 ==========")

        if os.path.exists(X_TRAIN_FILE) and os.path.exists(X_TEST_FILE):
            self.log(f"[提示] 检测到 {X_TRAIN_FILE} 和 {X_TEST_FILE} 已存在。")
            self.log("        如需重新提取，请先删除这两个文件再点此按钮。")
            return

        if not os.path.exists(AAL_PATH):
            self.log(f"[错误] 找不到 AAL3 图谱：{AAL_PATH}")
            return

        from nilearn.datasets import load_mni152_template
        from nilearn.image import resample_to_img, load_img
        from nilearn.maskers import NiftiLabelsMasker

        self.log("加载标准脑模板（MNI152）...")
        mni = load_mni152_template(resolution=2)

        self.log(f"加载 AAL3 脑图谱：{AAL_PATH}")
        masker = NiftiLabelsMasker(
            labels_img=AAL_PATH,
            strategy="mean",
            standardize=None
        )

        labels = pd.read_csv(LABEL_FILE, encoding='utf-8-sig')
        train_ids = labels['subject_id'].tolist()
        y = labels['label'].values

        test_files = sorted(os.listdir(TEST_DIR))
        test_ids = [f.replace('_T1w.nii.gz', '') for f in test_files]

        self.log(f"开始提取训练集特征（共 {len(train_ids)} 例）...")
        train_features = []
        for i, sid in enumerate(train_ids):
            path = os.path.join(TRAIN_DIR, f"{sid}_T1w.nii.gz")
            img = load_img(path)
            img_resampled = resample_to_img(img, mni, interpolation='continuous')
            feat = masker.fit_transform(img_resampled)
            train_features.append(feat.flatten())
            if (i + 1) % 10 == 0:
                self.log(f"  训练集进度：{i + 1}/{len(train_ids)}")

        self.log(f"开始提取测试集特征（共 {len(test_files)} 例）...")
        test_features = []
        for i, f in enumerate(test_files):
            path = os.path.join(TEST_DIR, f)
            img = load_img(path)
            img_resampled = resample_to_img(img, mni, interpolation='continuous')
            feat = masker.transform(img_resampled)
            test_features.append(feat.flatten())
            self.log(f"  测试集进度：{i + 1}/{len(test_files)}")

        X_train = pd.DataFrame(train_features)
        X_train['label'] = y
        X_train.to_csv(X_TRAIN_FILE, index=False)

        X_test = pd.DataFrame(test_features)
        X_test['subject_id'] = test_ids
        X_test.to_csv(X_TEST_FILE, index=False)

        self.log(f"特征提取完成，训练集形状：{X_train.shape}")
        self.log(f"特征提取完成，测试集形状：{X_test.shape}")
        self.log("========== 特征提取完毕 ==========")

    # ==================== 任务3：训练模型 ====================
    def task_train(self):
        self.log("========== 开始训练模型 ==========")

        if not (os.path.exists(X_TRAIN_FILE) and os.path.exists(X_TEST_FILE)):
            self.log(f"[错误] 未找到特征文件，请先执行“2. 提取特征”。")
            return

        train_data = pd.read_csv(X_TRAIN_FILE)
        X_train = train_data.drop('label', axis=1)
        y_train = train_data['label'].values

        self.clear_metric()
        self.clear_prediction()
        self.set_metric(f"训练集样本数：{X_train.shape[0]}，特征维度：{X_train.shape[1]}")

        for r in range(N_ROUNDS):
            round_num = r + 1
            rs = round_num * 10

            # 根据轮次选择不同的模型
            if round_num == 1:
                clf = SVC(kernel='rbf', class_weight='balanced',
                          probability=True, random_state=rs)
                model_name = "SVM"
            elif round_num == 2:
                clf = LogisticRegression(class_weight='balanced',
                                         max_iter=1000, random_state=rs)
                model_name = "逻辑回归"
            else:
                clf = RandomForestClassifier(n_estimators=100,
                                             class_weight='balanced',
                                             random_state=rs)
                model_name = "随机森林"

            self.log(f"---------- 训练第 {round_num} 个模型（{model_name}）----------")

            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', clf)
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
            y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)

            acc = accuracy_score(y_train, y_pred_cv)
            cm = confusion_matrix(y_train, y_pred_cv, labels=[1, 2])
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

            self.log(f"  模拟考总正确率: {acc:.4f}")
            self.log(f"  自闭症识别率  : {sensitivity:.4f}")
            self.log(f"  健康人识别率  : {specificity:.4f}")
            self.set_metric(
                f"第{round_num}次({model_name}) | ACC={acc:.4f}  Sens={sensitivity:.4f}  Spec={specificity:.4f}")

            # 用全部训练集训练最终模型
            model.fit(X_train, y_train)

            # 保存模型
            with open(MODEL_FILES[r], 'wb') as f:
                pickle.dump(model, f)
            self.log(f"  已保存模型：{MODEL_FILES[r]}")

        self.log("========== 三个模型全部训练完毕 ==========")

    # ==================== 任务4：预测测试集 ====================
    def task_predict(self):
        self.log("========== 开始预测测试集 ==========")
        self.clear_prediction()

        if not os.path.exists(X_TEST_FILE):
            self.log(f"[错误] 找不到测试集特征文件：{X_TEST_FILE}")
            return

        for mf in MODEL_FILES:
            if not os.path.exists(mf):
                self.log(f"[错误] 找不到模型文件：{mf}，请先执行“3. 训练模型”。")
                return

        test_data = pd.read_csv(X_TEST_FILE)
        test_ids = test_data['subject_id'].values
        X_test = test_data.drop('subject_id', axis=1)

        model_names = ['SVM', '逻辑回归', '随机森林']

        for r in range(N_ROUNDS):
            round_num = r + 1
            self.log(f"---------- 使用第 {round_num} 个模型（{model_names[r]}）预测 ----------")

            with open(MODEL_FILES[r], 'rb') as f:
                model = pickle.load(f)

            probs = model.predict_proba(X_test)
            asd_probs = probs[:, 0]  # 属于ASD(标签1)的概率

            # 按ASD概率从高到低排序，前5个填1，后5个填2
            sorted_indices = np.argsort(asd_probs)[::-1]
            final_labels = np.ones(len(test_ids), dtype=int)
            final_labels[sorted_indices[5:]] = 2

            submission = pd.DataFrame({
                'subject_id': test_ids,
                'label': final_labels
            })
            submission = submission.set_index('subject_id').loc[test_ids].reset_index()
            submission.to_csv(SUBMISSION_FILES[r], index=False)

            # 输出到界面右下角
            self.set_prediction(f"--- 第 {round_num} 次（{model_names[r]}）---")
            for _, row in submission.iterrows():
                self.set_prediction(f"{row['subject_id']} -> {row['label']}")
            self.set_prediction("")

            self.log(f"  预测结果已保存为 {SUBMISSION_FILES[r]}")

        self.log("========== 三次预测全部完成 ==========")
        self.log("请在项目文件夹查看 submission_1.csv / submission_2.csv / submission_3.csv")


# ==================== 启动入口 ====================
if __name__ == '__main__':
    root = tk.Tk()
    app = ASDHCApp(root)
    root.mainloop()