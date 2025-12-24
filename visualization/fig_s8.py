import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'
    means = [
        np.array([
            [0.98575, 0.01247, 0.00179],
            [0.01078, 0.98029, 0.00893],
            [0.00000, 0.00714, 0.99286]
        ]), 
        np.array([
            [0.98571, 0.01250, 0.00179],
            [0.01084, 0.96747, 0.02169],
            [0.00357, 0.02146, 0.97497]
        ]),
        np.array([
            [0.96951, 0.01971, 0.01078],
            [0.00893, 0.95708, 0.03399],
            [0.01435, 0.06263, 0.92302]
        ]),
    ]

    stds = [
        np.array([
            [0.02230, 0.01794, 0.00536],
            [0.01198, 0.02033, 0.01830],
            [0.00000, 0.00875, 0.00875]
        ]),
        np.array([
            [0.01557, 0.01395, 0.00536],
            [0.01192, 0.02095, 0.01347],
            [0.00714, 0.02082, 0.02550]
        ]),
        np.array([
            [0.01786, 0.01684, 0.01188],
            [0.01440, 0.02900, 0.02456],
            [0.01751, 0.03685, 0.04154]
        ])
    ]

    titles = ["DEP-Induced (Center)", "DEP-Induced (Radial)", "No Voltage"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), sharex = True, sharey = True)

    cmap = sns.color_palette("flare_r", as_cmap=True)
    vmin, vmax = 0, 100

    for ax, mean, std, title in zip(axes, means, stds, titles):
        sns.heatmap(
            mean*100, 
            ax = ax,
            vmin=vmin, vmax=vmax,
            fmt="", 
            cmap=cmap,
            cbar = False,
            linewidths=0.5,
            linecolor="gray",
        )
        fig.subplots_adjust(bottom=0.2)

        # Add text annotations manually
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                main_text = f"{mean[i, j]*100:.2f}%"
                std_text = f"(±{std[i, j]*100:.2f}%)"

                ax.text(
                    j + 0.5, i + 0.45, main_text,
                    ha='center', va='center',
                    fontsize=20, fontweight='bold', color='black' if mean[i, j] > 0.5 else 'white'
                )
                ax.text(
                    j + 0.5, i + 0.60, std_text,
                    ha='center', va='center',
                    fontsize=18, color='black' if mean[i, j] > 0.5 else 'white'
                )

        ax.set_xticklabels(["Normal", "Lung\nCancer", "Pancreatic\nCancer"], ha='center', fontsize=18)
        ax.set_yticklabels(["Normal", "Lung\nCancer", "Pancreatic\nCancer"], va='center', ha='center', fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.tick_params(labelleft=True)
        ax.tick_params(axis='y', pad=18)

    fig.text(0.5, 0.07, 'Predicted label', ha='center', fontsize=20, fontweight='bold')
    fig.text(0.07, 0.5, 'True label', va='center', rotation='vertical', fontsize=20, fontweight='bold')

    cbar = fig.colorbar(axes[0].collections[0], ax = axes, orientation = 'vertical', fraction=0.02, pad=0.03)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Percentage (%)', fontsize = 18)
    plt.savefig('fig_s8.png', dpi=300)
    plt.show()