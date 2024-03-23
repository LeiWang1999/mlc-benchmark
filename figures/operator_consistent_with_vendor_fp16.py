import matplotlib.pyplot as plt
import numpy as np

gemm_provider = ["M0","M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","M11","M12"]

gemm_times_data = [
    ('cuBLAS', (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, )),
    ('CUTLASS-GTX4090', (1.030645444, 1.033231209, 1.18256941, 1.02803345, 1.022267277, 1.265114196, 0.996555159, 1.009163621, 1.079376451, 1.011332969, 1.134750425, 0.990249653, 0.993835188)),
    ('BitBLAS-GTX4090', (1.036415874, 1.044765388, 1.202394974, 1.033710242, 1.013289806, 1.166284404, 0.994501833, 1.076019428, 1.030296473, 1.024340334, 1.048916227, 1.065301225, 1.049059481)),
    ('CUTLASS-A100', (1.020916611, 0.945489705, 0.977137685, 1.003658708, 1.301038347, 0.940134476, 0.938944645, 0.914001186, 0.978167206, 0.991563087, 0.979707351, 0.937908068, 1.007475496)),
    ('BitBLAS-A100', (0.93739573, 0.875446407, 0.921888922, 0.925103617, 1.195374786, 0.797576155, 0.880384494, 0.864692687, 0.97814059, 0.894402347, 0.894984695, 0.865421395, 1.014247426)),
]

providers = gemm_provider
times_data = gemm_times_data
num_ops = 8
providers = providers[:num_ops]
for i in range(len(times_data)):
    times_data[i] = (times_data[i][0], times_data[i][1][:num_ops])
_1x_baseline = "cuBLAS"
_1x_baseline_times = dict(times_data)[_1x_baseline]

# 计算其他方法相对加速比
speed_up_data = []
for label, times in times_data:
    if label != _1x_baseline:
        speed_up_data.append((label, times))

print(speed_up_data)
# Data
colers_sets = [
    # nilu
    # (20 / 255, 54 / 255, 95 / 255),
    # (118 / 255, 162 / 255, 185 / 255),
    # (191 / 255, 217 / 255, 229 / 255),
    # (214 / 255, 79 / 255, 56 / 255),
    # (112 / 255, 89 / 255, 146 / 255),
    # dori
    (169 / 255, 115 / 255, 153 / 255),
    (248 / 255, 242 / 255, 236 / 255),
    (214 / 255, 130 / 255, 148 / 255),
    (243 / 255, 191 / 255, 202 / 255),
    # coller
    (124 / 255, 134 / 255, 65 / 255),
    (185 / 255, 198 / 255, 122 / 255),
    (248 / 255, 231 / 255, 210 / 255),
    (182 / 255, 110 / 255, 151 / 255),
]
hatch_patterns = ["x", "\\", "*", "o", "O", ".", "-", "+"]

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.16

# Plotting
fig, ax = plt.subplots(figsize=(16, 5))
x = np.arange(len(providers))

# Draw cublas as a horizontal dashed line
ax.axhline(y=1, color="black", linestyle="dashed", label="cuBLAS")


# Draw a vertical dashed line to separate the two data parts
def get_inverse(a):
    inverse = [1 / i for i in a]
    return inverse

# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    rec = ax.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=hatch_patterns[i % 8],
        color=colers_sets[i],
    )

# set y-limit
max_speedup = max([max(speedup) for _, speedup in speed_up_data])
ax.set_ylim(0, max_speedup * 1.2)
# 调整图例位置和大小
legend_fontsize = 14

handles, labels = ax.get_legend_handles_labels()

# 将图例放置在图表中间
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=(len(labels) + 1) // 2 + 1,
    fontsize=legend_fontsize,
    frameon=False,
)
# X-axis and labels
ax.set_xlabel("Shapes from LLM", fontsize=18)
ax.set_ylabel("Speedup vs cuBLAS", fontsize=20)
ax.set_xticks(x + len(speed_up_data) * bar_width / len(times_data))
ax.set_xticklabels(providers)
ax.grid(axis="y", linestyle="--", linewidth=0.5)

# disable grid
ax.grid(False)

# add a title
plt.title("Speedup of GEMM on A100 and GTX 4090 (FP16)", fontsize=18)

# Save the plot to a file
plt.savefig("pdf/op_benchmark_consistent_gemm_fp16.pdf")
plt.savefig("png/op_benchmark_consistent_gemm_fp16.png", bbox_inches='tight', transparent=False, dpi=150)
