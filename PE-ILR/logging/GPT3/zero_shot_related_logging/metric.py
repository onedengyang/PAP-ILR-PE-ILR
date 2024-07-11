#coding=utf-8
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score,f1_score,roc_auc_score,matthews_corrcoef,confusion_matrix

# 获取当前目录下所有.csv文件
file_list = [f for f in os.listdir('.') if f.endswith('.csv')]

# 读取所有csv文件并合并
dfs = [pd.read_csv(f) for f in file_list]
merged_df = pd.concat(dfs)

# 提取label列和pro列
labels = merged_df['label']
predictions = merged_df['pro']

# 计算精确率和召回率
import numpy as np
eval_acc = np.mean(labels == predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f_measure = f1_score(labels, predictions)
eval_auc = roc_auc_score(labels, predictions)
eval_mcc = matthews_corrcoef(labels, predictions)
tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=predictions).ravel()
eval_pf = fp / (fp + tn)
result = {

    "eval_acc": round(float(eval_acc), 4),
    "eval_precision": round(precision, 4),
    "eval_recall": round(recall, 4),
    "eval_f1": round(f_measure, 4),
    "eval_auc": round(eval_auc, 4),
    "eval_mcc": round(eval_mcc, 4),
    "eval_pf": round(eval_pf, 4),
}
import pandas as pd
df = pd.DataFrame([result])
print(result)
df.to_csv("log_zero_shot_result.csv")