import os
import matplotlib.pyplot as plt
import numpy as np

results_dir = "results"
experiments = ["cpu"]
data = {exp: {"openmp": {}, "starpu": {}} for exp in experiments}
experiments_nbodies_start = 12
experiments_nbodies_end = 17
experiments_nruns = 3

for exp in experiments:
    for tool in ["openmp", "starpu"]:
        prefix = f"{tool}_{exp}-"
        for folder in [prefix + str(i) for i in range(experiments_nbodies_start, experiments_nbodies_end + 1)]:
            folderpath = os.path.join(results_dir, folder)
            times = []
            for file in [str(i) for i in range(1, experiments_nruns + 1)]:
                with open(os.path.join(folderpath, file), "r") as f:
                    times.append(float(f.read().strip()))

            avg_time = sum(times) / len(times)
            if tool == "starpu":
                avg_time /= 1_000_000  # Convert to seconds

            data[exp][tool][folder[-2:]] = avg_time

os.makedirs("plots", exist_ok=True)
for exp in experiments:
    keys = sorted(data[exp]["openmp"].keys())  # Sorted molecule counts
    openmp_values = [data[exp]["openmp"][k] for k in keys]
    starpu_values = [data[exp]["starpu"][k] for k in keys]

    x = np.arange(len(keys))  # Label positions
    width = 0.4  # Increased bar width to reduce spacing

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, openmp_values, width, label="OpenMP")
    plt.bar(x + width / 2, starpu_values, width, label="StarPU")

    # Add text labels
    for i, (o, s) in enumerate(zip(openmp_values, starpu_values)):
        plt.text(x[i] - width / 2, o, f"{o:.2f}", ha="center", va="bottom", fontsize=10)
        plt.text(x[i] + width / 2, s, f"{s:.2f}", ha="center", va="bottom", fontsize=10)

    keys = map(lambda x: "%d" % (int(x) + 1), keys)
    plt.xticks(x, keys)
    plt.xlabel("Number of Particles (2^n)")
    plt.ylabel("Execution Time (s)")
    plt.title(f"OpenMP vs StarPU - {exp.replace('_', ' ').upper()}")
    
    # Adjusted legend placement at bottom
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Ensure space for legend at the bottom

    plt.savefig(f"plots/{exp}.png")
    plt.close()

print("Plots saved")