import os
import json
import pandas as pd

# 指定包含所有模型输出文件夹的根目录
base_path = "logs_and_outputs/long/outputs_Olora_SAM-020"

# 模型文件夹名称顺序（即每一行）
model_folders = [
    "1-yelp", "2-amazon", "3-MNLI", "4-CB", "5-COPA", "6-QQP", "7-RTE",
    "8-IMDB", "9-SST-2", "10-dbpedia", "11-agnews", "12-yahoo",
    "13-MultiRC", "14-BoolQA", "15-WiC"
]

# 数据集名称（即每一列）
dataset_order = [
    "yelp", "amazon", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2",
    "dbpedia", "agnews", "yahoo", "MultiRC", "BoolQA", "WiC"
]

# 存储提取的结果
results = {}

for folder in model_folders:
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
df = df.reindex(columns=dataset_order)

# 添加平均值
# 添加每个模型（行）的平均值
df["Model_Average"] = df.mean(axis=1)
# 添加每个数据集（列）的平均值
df.loc["Data_Average"] = df.mean(axis=0)

# 重置索引使模型名成为列
df.index.name = "↘ Model ↓ / Test →"
df.reset_index(inplace=True)

# 保存为 Excel

output_file = os.path.join(base_path, "merge_result.xlsx")
df.to_excel(output_file, index=False)

