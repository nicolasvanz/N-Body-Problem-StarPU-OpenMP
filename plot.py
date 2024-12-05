import os
import matplotlib.pyplot as plt

results_dir = "results"
experiment = "openmp_cpu-"
data = {}

for experiment in ["openmp_cpu", "openmp_gpu", "openmp_cpu_gpu", "starpu_cpu", "starpu_gpu", "starpu_cpu_gpu"]:
    experiment+="-"
    for folder in [experiment+str(i) for i in range(11,18)]:
        folderpath = os.path.join(results_dir, folder)
        for file in [str(i) for i in range(1, 8)]:
            times = []
            with open(os.path.join(folderpath, file), "r") as f:
                times.append(float(f.read().strip()))
        data[folder[-2:]] = sum(times) / len(times)
        if experiment.startswith("starpu"):
            data[folder[-2:]] /= 1000000

    plt.figure(figsize=(12, 6))
    plt.plot(data.keys(), data.values())
    for i,j in zip(data.keys(),data.values()):
        plt.annotate('%.2f' % j, xy=(i,j))

    plt.title(" ".join(experiment[:-1].split("_")))
    plt.xlabel("number of molecules (2^n)")
    plt.ylabel("Execution Time (s)")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig("plots/%s" % experiment[:-1])
