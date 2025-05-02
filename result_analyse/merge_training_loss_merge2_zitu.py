import os
import json
import matplotlib.pyplot as plt

# 模型文件夹名称按顺序排列
model_folders_order1 = ["1-dbpedia", "2-amazon", "3-yahoo", "4-agnews"]
test_dataset_order = ["dbpedia", "amazon", "yahoo", "agnews"]

# 两组模型所在路径
# base_path_1 = "logs_and_outputs_showResults/order_1/Olora_T5largeSAM_torch"
# base_path_2 = "logs_and_outputs_showResults/order_1/Olora_T5large_torch"

base_path_2 = "logs_and_outputs_showResults/order_1/Inclora_T5large_torch_AdamwHF"
base_path_1 = "logs_and_outputs_showResults/order_1/Inclora_T5large_torch_SAMAdamwHF"

# 存储结果字典
results = {}

# 定义读取函数
def collect_loss_curves(base_path, label_prefix):
    local_results = {}
    for folder, dataset in zip(model_folders_order1, test_dataset_order):
        trainer_state_path = os.path.join(base_path, folder, "trainer_state.json")
        if os.path.isfile(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                data = json.load(f)
            log_history = data.get("log_history", [])
            steps = [entry["step"] for entry in log_history if "loss" in entry]
            losses = [entry["loss"] for entry in log_history if "loss" in entry]
            local_results[f"{label_prefix}-{dataset}"] = (steps, losses)
    return local_results

# 收集两组结果
results.update(collect_loss_curves(base_path_1, "SAM"))
results.update(collect_loss_curves(base_path_2, "AdamW"))

# 创建一个包含4个子图的图像（2x2布局）
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()  # 转换为一维方便索引

# 数据集顺序保证与子图索引一致
datasets = ["dbpedia", "amazon", "yahoo", "agnews"]

# 绘制每个子图
for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    for optimizer in ["SAM", "AdamW"]:
        label = f"{optimizer}-{dataset}"
        if label in results:
            steps, losses = results[label]
            linestyle = "-" if optimizer == "SAM" else "--"
            ax.plot(steps, losses, label=optimizer, linestyle=linestyle, linewidth=1.5)
    
    ax.set_title(f"Dataset: {dataset}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.grid(True)
    ax.legend()

plt.suptitle("Training Loss Comparison (SAM vs AdamW) per Dataset", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 保存 PDF 文件
pdf_path = os.path.join(base_path_1, "loss_curves_per_dataset_subplots.pdf")
plt.savefig(pdf_path)

print(f"Saving PDF..in {pdf_path}")

