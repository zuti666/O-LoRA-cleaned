import os
import json
import matplotlib.pyplot as plt

# 模型文件夹名称按顺序排列
model_folders_order1 = [
    "1-dbpedia", "2-amazon", "3-yahoo", "4-agnews"
]

# 数据集名称，用于图例显示
test_dataset_order = [
    "dbpedia", "amazon", "yahoo", "agnews",
]

# 假设 base_path 为当前路径下的子目录
# base_path = "logs_and_outputs_showResults/order_1/Nlora_T5large_torch_SAMAdamwHF005-10"
base_path_1 = "logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10"


# 初始化结果存储字典
results_1 = {}

# 遍历每个模型文件夹，读取 trainer_state.json 文件中的 loss 曲线
for folder, dataset in zip(model_folders_order1, test_dataset_order):
    trainer_state_path = os.path.join(base_path_1, folder, "trainer_state.json")
    if os.path.isfile(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            data = json.load(f)
        log_history = data.get("log_history", [])
        steps = [entry["step"] for entry in log_history if "loss" in entry]
        losses = [entry["loss"] for entry in log_history if "loss" in entry]
        results_1[dataset] = (steps, losses)

# 绘图
plt.figure(figsize=(10, 6))

for dataset, (steps, losses) in results_1.items():
    plt.plot(steps, losses, label=dataset)


plt.xlabel("Training Step")
plt.ylabel("Training Loss")
plt.title("Training Loss Curves for Different Datasets")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存为 PDF
pdf_path = os.path.join(base_path_1, "loss_curves_comparison.pdf")
plt.savefig(pdf_path)


