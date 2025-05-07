import os
import matplotlib.pyplot as plt
import numpy as np

results_dir = "results"
experiments = ["cpu"]
data = {exp: {"c5n_starpu": {}, "m5_starpu": {}} for exp in experiments}
experiments_nbodies_start = 18
experiments_nbodies_end = 19
experiments_nruns = 3
plot_font_size=20

for exp in experiments:
    for tool in ["c5n_starpu", "m5_starpu"]:
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
    keys = sorted(data[exp]["c5n_starpu"].keys())  # Sorted molecule counts
    openmp_values = [data[exp]["c5n_starpu"][k] for k in keys]
    starpu_values = [data[exp]["m5_starpu"][k] for k in keys]

    x = np.arange(len(keys))  # Label positions
    width = 0.4  # Increased bar width to reduce spacing

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, openmp_values, width, label="c5n.18xlarge")
    plt.bar(x + width / 2, starpu_values, width, label="m5.16xlarge")

    # Add text labels
    for i, (o, s) in enumerate(zip(openmp_values, starpu_values)):
        plt.text(x[i] - width / 2, o, f"{o:.2f}", ha="center", va="bottom", fontsize=plot_font_size)
        plt.text(x[i] + width / 2, s, f"{s:.2f}", ha="center", va="bottom", fontsize=plot_font_size)

    keys = map(lambda x: "%d" % (int(x) + 1), keys)
    plt.ylim(0, max(starpu_values) * 1.1)
    plt.xticks(x, keys)
    plt.xlabel("Number of Particles (2^n)")
    plt.ylabel("Execution Time (s)")
    # plt.title(f"N-Body Simulation Execution Time on CPU-Only Environments")
    
    # Adjusted legend placement at bottom
    # plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.legend(loc="upper left", ncol=1)
    
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(pad=0)

    plt.savefig(f"plots/{exp}.png", bbox_inches="tight")
    plt.close()

print("Plots saved")