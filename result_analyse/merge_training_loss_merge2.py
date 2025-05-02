import os
import json
import matplotlib.pyplot as plt

# 模型文件夹名称按顺序排列
model_folders_order1 = ["1-dbpedia", "2-amazon", "3-yahoo", "4-agnews"]
test_dataset_order = ["dbpedia", "amazon", "yahoo", "agnews"]

# 两组模型所在路径
base_path_1 = "logs_and_outputs_showResults/order_1/Nlora_T5large_torch_SAMAdamwHF005-10"
base_path_2 = "logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10"

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

# 绘图
# plt.figure(figsize=(12, 7))
# for label, (steps, losses) in results.items():
#     plt.plot(steps, losses, label=label)
# 颜色设定：SAM 用暖色，AdamW 用冷色
# warm_colors = ["orangered", "darkorange", "gold", "tomato"]
# cool_colors = ["blue", "steelblue", "teal", "navy"]

# # 绘图
# plt.figure(figsize=(12, 7))
# for idx, (label, (steps, losses)) in enumerate(results.items()):
#     if label.startswith("SAM"):
#         color = warm_colors[idx % 4]
#     else:
#         color = cool_colors[idx % 4]
#     plt.plot(steps, losses, label=label, color=color, linewidth=1)  # 设置更细线条


# 重新绘图：相同数据集使用相同颜色，不同组使用不同线型
plt.figure(figsize=(12, 7))

# 定义颜色映射：每个数据集固定一个颜色
color_map = {
    "dbpedia": "tab:blue",
    "amazon": "tab:orange",
    "yahoo": "tab:green",
    "agnews": "tab:red"
}

# 定义线型：不同组（优化器）使用不同线型
linestyle_map = {
    "SAM": "-",
    "AdamW": "--"
}

# 绘图循环
for label, (steps, losses) in results.items():
    optimizer, dataset = label.split("-")
    color = color_map[dataset]
    linestyle = linestyle_map[optimizer]
    plt.plot(steps, losses, label=label, color=color, linestyle=linestyle, linewidth=1.5)


plt.xlabel("Training Step")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison: SAM vs AdamW (T5-Large)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存 PDF 文件

pdf_path = os.path.join(base_path_1, "loss_curves_comparison_optimized.pdf")
plt.savefig(pdf_path)
print(f"Saving PDF..in {pdf_path}")

