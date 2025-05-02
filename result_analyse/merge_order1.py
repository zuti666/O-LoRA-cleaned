import os
import json
import pandas as pd
import numpy as np

# 指定包含所有模型输出文件夹的根目录
# base_path = "logs_and_outputs/order_1/outputs_redoOlora_T5large_torch"
# base_path = "logs_and_outputs/order_1/outputs_Olora_T5BaseSAM_torch"
# base_path ="logs_and_outputs_ShowResults/order_1/outputs_redoOlora_T5Base_torch"
# base_path ="logs_and_outputs/order1_olora/outputs"
# base_path ="logs_and_outputs_ShowResults/order_1/outputs_Olora_T5Base_torch_SAM010"

# ----- Inclora_T5large_torch_SAMAdamwHF
# base_path = "logs_and_outputs_showResults/order_1/Inclora_T5large_torch_SAMAdamwHF"
# base_path = "logs_and_outputs_showResults/order_1/Inclora_T5large_torch_AdamwHF"

# base_path ="logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHF"
# base_path = "logs_and_outputs_showResults/order_1/Nlora_T5large_torch_SAMAdamwHF"

# base_path ="logs_and_outputs_showResults/order_1/Inclora_T5large_torch_SAMAdamwHF-015"

# base_path ="logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10"

# base_path ="logs_and_outputs_showResults/order_1/Inclora_T5large_torch_AdamwHF-E10"


# base_path ="logs_and_outputs_showResults/order_1/Nlora_T5large_torch_SAMAdamwHF010-10"

# base_path ="logs_and_outputs_showResults/order_1/Nlora_T5large_torch_SAMAdamwHF005-10"

# base_path ="logs_and_outputs_showResults/order_1/Nlora_T5large_torch_SAMAdamwHF010-10"

# base_path = "logs_and_outputs_showResults/order_1/Nlora_T5large_torch_SAMAdamwHF005-1"

base_path = "logs_and_outputs_showResults/order_1/Nlora_T5large_torch_SAMAdamwHF002-1"

# 模型文件夹名称顺序（即每一行）
# model_folders = [
#     "1-yelp", "2-amazon", "3-MNLI", "4-CB", "5-COPA", "6-QQP", "7-RTE",
#     "8-IMDB", "9-SST-2", "10-dbpedia", "11-agnews", "12-yahoo",
#     "13-MultiRC", "14-BoolQA", "15-WiC"
# ]
model_folders_order1 = [
    "1-dbpedia", "2-amazon", "3-yahoo", "4-agnews"
]

# 数据集名称（即每一列）
test_dataset_order = [
    "dbpedia", "amazon", "yahoo", "agnews",
]

# 存储提取的结果
results = {}

for folder in model_folders_order1:
    model_path = os.path.join(base_path, folder, "all_results.json")
    print(f"Processing {model_path}...")
    if os.path.isfile(model_path):
        with open(model_path, "r") as f:
            data = json.load(f)
        model_results = {}
        for key, value in data.items():
            if key.startswith("predict_exact_match_for_"):
                dataset = key.replace("predict_exact_match_for_", "")
                model_results[dataset] = value
        results[folder] = model_results

# 构建DataFrame
df = pd.DataFrame.from_dict(results, orient='index')
df = df.reindex(columns=test_dataset_order)

# 添加平均值
# 添加每个模型（行）的平均值
df["Model_Average"] = df.mean(axis=1)
# 添加每个数据集（列）的平均值
df.loc["Data_Average"] = df.mean(axis=0)

# ---- 添加新的指标：Acc, AAA, Transfer, Last, FWT, BWT, Std ----
# 去除Data_Average行，只对模型行做处理
df.index.name = "↘ Model ↓ / Test →"
df.reset_index(inplace=True)

model_names = df.iloc[:-1]["↘ Model ↓ / Test →"].tolist()
df_metrics = df.set_index("↘ Model ↓ / Test →").drop(index="Data_Average")

# 只提取矩阵数值部分
matrix = df_metrics[test_dataset_order].values
N = matrix.shape[0]

print("Matrix", matrix)
print("Matrix shape:", matrix.shape)
# ---- 计算指标 ----

# 基本指标
DiagAvg = np.trace(matrix) / N
LastAvg = np.mean(matrix[-1, :])

# Transfer (右上三角)
transfer_scores = []
for i in range(N-1):
    for j in range(i+1, N):
        transfer_scores.append(matrix[i, j])
Transfer = np.mean(transfer_scores)


# Average
Average = (Transfer + LastAvg) / 2

# AAA (左下角含对角线)
aaa_scores = []
for i in range(N):
    for j in range(i+1):
        aaa_scores.append(matrix[i, j])
AAA = np.mean(aaa_scores)

# 新增指标
random_guess = 0  # 简化假设分类随机猜中为0，可根据需要调整

# FWT (Forward Transfer)
fwt_scores = []
for i in range(1, N):
    fwt_scores.append(matrix[i-1, i] - random_guess)
FWT = np.mean(fwt_scores)

# BWT (Backward Transfer)
bwt_scores = []
for i in range(N-1):
    bwt_scores.append(matrix[N-1, i] - matrix[i, i])
BWT = np.mean(bwt_scores)

# ---- 新增遗忘指标 ----
# FM (Forgetting Measure)
max_acc_per_task = np.max(matrix[:-1, :], axis=0)
FM = np.mean(max_acc_per_task[:-1] - matrix[-1, :-1])

# AUF (Area Under Forgetting Curve)
auf_list = []
for task_id in range(N-1):
    trajectory = matrix[task_id:, task_id]  # 从学到该任务后每一轮表现
    diffs = trajectory[:-1] - trajectory[1:]
    auf_list.append(np.sum(diffs))
AUF = np.mean(auf_list)

# RA (Retained Accuracy)：去除对角线的左下角平均
lower_triangular_scores = []
for i in range(1, N):
    lower_triangular_scores.extend(matrix[i, :i].tolist())
RA = np.mean(lower_triangular_scores)


# Std (标准差，对角线)
diagonal = np.diag(matrix)
Std_diag = np.std(diagonal)

# ---- 保存到Excel ----
# 创建新的指标DataFrame
new_metrics = pd.DataFrame({
    "Metric": [ "LastAvg","DiagAvg", "AAA", "Average","Std_diag","Transfer", "FWT", "FM","AUF","RA","BWT" ],
    "Value":  [  LastAvg, DiagAvg, AAA, Average,       Std_diag,Transfer,  FWT, FM,AUF,RA,BWT]
})

# 保存至新的Sheet中
output_file = os.path.join(base_path, "order1_torch_AdamW_merge_result.xlsx")
with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, index=False, sheet_name="Main_Result")
    new_metrics.to_excel(writer, index=False, sheet_name="Metrics")
