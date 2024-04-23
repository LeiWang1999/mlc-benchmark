import matplotlib.pyplot as plt
import numpy as np
from data.a100_gemm import s2_gemm_provider as providers
from data.a100_gemm import s2_gemm_speedup_data as speed_up_data

colormap = plt.cm.tab20b  # LinearSegmentedColormap

# Data
colers_sets = [
    # nilu
    (20 / 255, 54 / 255, 95 / 255),
    (118 / 255, 162 / 255, 185 / 255),
    (191 / 255, 217 / 255, 229 / 255),
    (214 / 255, 79 / 255, 56 / 255),
    (112 / 255, 89 / 255, 146 / 255),
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
marker_patterns = ["o", "^", "s", "P", "*", "+", "x", "D"]


# Create an array for x-axis positions
x = np.arange(len(providers))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(providers))

ax.axhline(y=1, color="black", linestyle="dashed", label="cuBLAS-W$_{FP16}$A$_{FP16}$")

# Create bars using a loop
for i, (label, speedup) in enumerate(speed_up_data):
    color = colormap((i  / len(speed_up_data)))
    rec = ax.plot(
        x,
        speedup,
        label=label,
        linewidth=2.0,
        marker=marker_patterns[i % 8],
        markersize=5,
        color=color
    )


# set y-limit
max_speedup = max([max(speedup) for _, speedup in speed_up_data])
ax.set_ylim(0, max_speedup * 1.2)
# 调整图例位置和大小
legend_fontsize = 11

handles, labels = ax.get_legend_handles_labels()

# 将图例放置在图表中间
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=(len(labels) + 1) // 3 + 1,
    fontsize=legend_fontsize,
    frameon=False,
)
# X-axis and labels
ax.set_xlabel("Shapes from LLM", fontsize=16)
ax.set_ylabel("Speedup of vs cuBLAS-W$_{FP16}$A$_{FP16}$", fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(providers)
ax.grid(axis="y", linestyle="--", linewidth=0.5)

# disable grid
ax.grid(False)

# add a title
plt.title(
    "Scaling Speedup of BitNET INT8xINT2 on A100",
    fontsize=16,
)
plt.style.use("ggplot")

# Save the plot to a file
plt.savefig("pdf/op_benchmark_a100_int2_scaling.pdf", bbox_inches='tight')
plt.savefig("png/op_benchmark_a100_int2_scaling.png", bbox_inches="tight", dpi=255)
