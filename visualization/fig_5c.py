import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'
    groups = 3
    bars_per_group = 3
    x = np.arange(groups)

    values = np.array([
        [98.029, 96.747, 95.708],
        [99.286, 97.497, 92.302],
        [98.575, 98.571, 96.951]
    ], dtype=float)

    errors = np.array([
        [2.143, 2.208, 3.057],
        [0.922, 2.688, 4.379],
        [2.350, 1.641, 1.882]
    ], dtype=float)

    colors = ['#df6262', '#69b180', '#679fcd']

    bar_width = 0.65

    plt.rcParams['font.family'] = 'Arial'
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
                bar.get_x() + bar.get_width() / 2,   # x-position (center of bar)
                height + err + 0.02,  # y-position (slightly above error bar)
                f'{val:.2f}',                        # text (formatted value)
                ha='center', va='bottom', fontsize=17, fontweight='bold'
            )

    ax.set_xticks(2.25*x + bar_width)
    ax.set_xticklabels([
        "Sensitivity\n(Lung Cancer)",
        "Sensitivity\n(Pancreatic Cancer)",
        "Specificity"
    ], fontweight='bold', fontsize = 18)

    ax.set_ylabel("Percentage (%)", fontsize = 18)
    plt.xlim(-0.5, 6.5)
    plt.ylim(87, 104)
    
    ax = plt.gca()
    ax.set_yticklabels(range(88, 101, 2), fontweight='bold', fontsize = 18)
    plt.yticks(range(88, 101, 2))

    ax.legend(
        fontsize=17,
        loc = 'upper center',
        frameon = False,
        borderaxespad = 0,
        columnspacing=1.45,
        ncol=3,
    )

    plt.tight_layout()
    plt.savefig("fig_5c.png", dpi=300)
    plt.show()