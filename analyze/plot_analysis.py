import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.core.fromnumeric import mean
import os

file_path = "all_persona_rewards.csv"
print("hi")


df = pd.read_csv(file_path, header=None)

means = df.mean(axis=0)

rw_mean = []
for i in range(0, len(means), 2):
    rw_mean.append(mean(means[i : i + 2]))
# print(rw_mean)

plt.bar(x=[i for i in range(30)], height=rw_mean)
plt.xlabel("persona")
plt.ylabel("score")
plt.title("avg over 25 cases")
plt.savefig("rw_avg.png")
plt.close()


for i_case in range(25):
    means = df.loc[i_case, :]
    rw_mean = []
    for i in range(0, len(means), 2):
        rw_mean.append(mean(means[i : i + 2]))

    plt.bar(x=[i for i in range(30)], height=rw_mean)
    plt.title(f"case {i_case}")
    plt.savefig(f"rw_case{i_case}.png")
    plt.close()
