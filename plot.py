import os
import matplotlib.pyplot as plt
import numpy as np

results_dir = "results"
experiments = ["cpu_gpu"]
data = {exp: {"starpu": {},} for exp in experiments}
experiments_nbodies_start = 18
experiments_nbodies_end = 19
experiments_nruns = 3
plot_font_size=20

for exp in experiments:
    for tool in ["starpu"]:
        prefix = f"{tool}_{exp}-"
        for folder in [prefix + str(i) for i in range(experiments_nbodies_start, experiments_nbodies_end + 1)]:
            folderpath = os.path.join(results_dir, folder)
            times = []
            for file in [str(i) for i in range(1, len(os.listdir(folderpath)) + 1)]:
                with open(os.path.join(folderpath, file), "r") as f:
                    times.append(float(f.read().strip()))

            avg_time = sum(times) / len(times)
            # if tool == "starpu":
            avg_time /= 1_000_000  # Convert to seconds

            data[exp][tool][folder[-2:]] = avg_time

plt.rcParams.update({'font.size': plot_font_size})  # Change 14 to any size you prefer

os.makedirs("plots", exist_ok=True)
for exp in experiments:
    keys = sorted(data[exp]["starpu"].keys())  # Sorted molecule counts
    starpu_values = [data[exp]["starpu"][k] for k in keys]

    # x = np.arange(len(keys))  # Label positions
    x = np.linspace(0, len(keys) - 1, len(keys)) * 0.5

    width = 0.4  # Increased bar width to reduce spacing

    plt.figure(figsize=(9, 7))
    plt.bar(x, starpu_values, width, label="g6.16xlarge")

    # Add text labels
    # for i, (o, s) in enumerate(zip(openmp_values, starpu_values)):
    #     plt.text(x[i] - width / 2, o, f"{o:.2f}", ha="center", va="bottom", fontsize=plot_font_size)
    #     plt.text(x[i] + width / 2, s, f"{s:.2f}", ha="center", va="bottom", fontsize=plot_font_size)
    for i, s in enumerate(starpu_values):
        plt.text(x[i], s + 0.01, f"{s:.2f}", ha="center", va="bottom", fontsize=plot_font_size)


    keys = map(lambda x: "%d" % (int(x) + 1), keys)
    max_val = max(starpu_values)
    plt.ylim(0, max_val * 1.1)
    plt.xticks(x, keys)
    plt.xlabel("Number of Particles (2^n)")
    plt.ylabel("Execution Time (s)")
    # plt.title(f"CPU+GPU Execution Time ")
    
    # Adjusted legend placement at bottom
    # plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.subplots_adjust(bottom=0.15)  # Ensure space for legend at the bottom
    
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(pad=0)

    plt.savefig(f"plots/{exp}.png", bbox_inches="tight")
    plt.close()

print("Plots saved")