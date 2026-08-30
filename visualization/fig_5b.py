import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'
    groups = 3
    bars_per_group = 3
    x = np.arange(groups)

    values = np.array([
        [98.629, 97.606, 94.982],
        [97.262, 97.064, 93.363],
        [97.260, 94.971, 92.286],
    ], dtype=float)

    errors = np.array([
        [0.894, 1.540, 1.668],
        [1.037, 1.328, 1.359],
        [1.168, 1.606, 1.673],
    ], dtype=float)

    colors = ['#df6262', '#69b180', '#679fcd']
    bar_width = 0.65

    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(bars_per_group):
        bars = ax.bar(
            2.25*x + i * bar_width,
            values[:, i],
            bar_width,
            yerr=errors[:, i],
            capsize=3,
            color=colors[i],
            edgecolor='black',
            label='DEP-Induced (Center)' if i == 0 else 'DEP-Induced (Radial)' if i == 1 else 'No Voltage'
        )

        # Add text labels on top of the error bars
        for bar, val, err in zip(bars, values[:, i], errors[:, i]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + err + 0.02,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=17, fontweight='bold'
            )

    ax.set_xticks(2.25*x + bar_width)
    ax.set_xticklabels([
        "3-CNN",
        "ResNet",
        "MLP"
    ], fontweight='bold', fontsize = 18)

    ax.set_ylabel("Accuracy (%)", fontsize = 18)
    plt.xlim(-0.5, 6.35)
    plt.ylim(89.25, 102)
    
    ax = plt.gca()
    ax.set_yticklabels(range(90, 101, 2), fontweight='bold', fontsize = 18)
    plt.yticks(range(90, 101, 2))

    ax.legend(
        fontsize=17,
        loc = 'upper center',
        frameon = False,
        borderaxespad = 0,
        columnspacing=1.45,
        ncol=3,
    )

    plt.tight_layout()
    plt.savefig("fig_5b.png", dpi=300)
    plt.show()