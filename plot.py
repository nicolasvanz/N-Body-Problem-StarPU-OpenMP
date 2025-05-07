import os
import matplotlib.pyplot as plt
import numpy as np

results_dir = "results"
experiments = ["gpu"]
data = {exp: {"2-starpu": {}, "4-starpu": {}, "8-starpu": {}} for exp in experiments}
experiments_nbodies_start = 18
experiments_nbodies_end = 19
plot_font_size=20

for exp in experiments:
    for tool in ["2-starpu", "4-starpu", "8-starpu"]:
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
plt.subplots_adjust(bottom=0.15)  # Ensure space for legend at the bottom

os.makedirs("plots", exist_ok=True)
for exp in experiments:
    keys = sorted(data[exp]["2-starpu"].keys())  # Sorted molecule counts
    values_2 = [data[exp]["2-starpu"][k] for k in keys]
    values_4 = [data[exp]["4-starpu"][k] for k in keys]
    values_8 = [data[exp]["8-starpu"][k] for k in keys]

    # x = np.arange(len(keys))  # Label positions
    x = np.linspace(0, len(keys) - 1, len(keys)) * 0.8
    width = 0.25  # Increased bar width to reduce spacing

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, values_2, width, label="2 nodes")
    plt.bar(x, values_4, width, label="4 nodes")
    plt.bar(x + width, values_8, width, label="8 nodes")

    # Add text labels
    for i, (o, s, t) in enumerate(zip(values_2, values_4, values_8)):
        plt.text(x[i] - width, o, f"{o:.2f}", ha="center", va="bottom", fontsize=plot_font_size)
        plt.text(x[i], s, f"{s:.2f}", ha="center", va="bottom", fontsize=plot_font_size)
        plt.text(x[i] + width, t, f"{t:.2f}", ha="center", va="bottom", fontsize=plot_font_size)

    keys = map(lambda x: "%d" % (int(x) + 1), keys)
    plt.ylim(0, max(values_2) * 1.1)
    plt.xticks(x, keys)
    plt.xlabel("Number of Particles (2^n)")
    plt.ylabel("Execution Time (s)")
    # plt.title(f"N-Body Simulation Execution Time Across Nodes")
    
    # Adjusted legend placement at bottom
    # plt.legend(loc="bottom center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.legend(loc="upper left", ncol=1)
    
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(pad=0)
    

    plt.savefig(f"plots/multinode.png", bbox_inches="tight")
    plt.close()

print("Plots saved")