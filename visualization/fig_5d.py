import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'
    
    voltages = ["DEP-Induced (Center)", "DEP-Induced (Radial)", "No Voltage"]
    ax_titles = ['Normal', 'Lung\nCancer', 'Pancreatic\nCancer']

    shifts = np.genfromtxt('shifts.csv', delimiter=',')

    fig, axes = plt.subplots(
        nrows=3, ncols=3, 
        figsize=(22, 6), 
        sharex=True, sharey=True,
        gridspec_kw={'height_ratios': [1,1,1]}
    )

    plt.subplots_adjust(wspace=0.12, hspace=0.3)

    for col, voltage in enumerate(["center", "surroundings", "novoltage"]):        
        data = []
        maxes = []
        mins = []
        for t in ['normal', 'lung', 'pan']:
            single_data = []
            for i in range(10):
                file_name = f'ig/{voltage}_3CNN_random_none_250_{i}outof10_attrs_{t}.csv'
                file_data = np.genfromtxt(file_name, delimiter=',')
                # print(file_data.shape)
                single_data.append(file_data)
            print(single_data)
            avg_data = np.mean(single_data, axis=0)
            data.append(avg_data)
            maxes.append(np.max(avg_data))
            mins.append(np.min(avg_data))

        for row, (single_data, ax_title) in enumerate(zip(data, ax_titles)):
            
            ax = axes[row, col]
            im = ax.imshow(np.tile(single_data, (675, 1)), cmap=sns.color_palette("vlag", as_cmap=True), aspect='auto', vmin=min(mins), vmax=max(maxes)) # cmap=sns.color_palette("vlag", as_cmap = True) or "icefire"
            ax.set_yticks([])
            if col == 0:  # Only put row titles on first column
                ax.set_ylabel(ax_title, fontsize=18)
            if row == 0:  # Top row gets column titles
                ax.set_title(voltages[col], fontsize=22, pad=10, fontweight='bold')
            if row == 2:
                ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=16)

        # Add one colorbar for all plots
        cbar = fig.colorbar(im, ax=axes[:, col].ravel().tolist(), label='Attributions', aspect=50, pad=0.085)
        if col == 0:
            from matplotlib import ticker
            cbar.locator = ticker.MultipleLocator(base=0.2)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Attributions', fontsize=16, labelpad=12 if col == 0 else 7 if col == 1 else 2)

    target_integers = range(600, 1801, 200)
    indices = [np.abs(shifts - target).argmin() for target in target_integers]
    indices[-1] = len(shifts)  # fixes tick alignment
    target_minor_integers = range(700, 1801, 200)
    minor_indices = [np.abs(shifts - target).argmin() for target in target_minor_integers]
    for ax in axes.reshape(-1):
        ax.set_xticks(indices)
        ax.set_xticks(minor_indices, minor = True)
        ax.set_xticklabels(np.arange(600, 1801, 200), fontsize=17)
        ax.tick_params(labelbottom=True)
    
    plt.savefig('fig_5d.png', dpi=300)
    plt.show()