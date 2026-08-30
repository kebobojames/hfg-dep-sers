import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.datatable import DataTableUrine1115
from ..utils.ipf import iterative_polynomial_fitting_2

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'

    novoltage_datatable = DataTableUrine1115("250807_urine_data/1.) No Voltage", cache_fold_path='./250807_urine_cache', prefix = 'novoltage')
    center_datatable = DataTableUrine1115("250807_urine_data/2.) Voltage Application (Center)", cache_fold_path='./250807_urine_cache', prefix = 'center')
    surroundings_datatable = DataTableUrine1115("250807_urine_data/3) Voltage Application (Surroundings)", cache_fold_path='./250807_urine_cache', prefix = 'surroundings')

    averages = []
    percentiles = []

    offset = range(72000, -1, -9000)
    for i, dt in enumerate([center_datatable, surroundings_datatable, novoltage_datatable]):
        X, y, shift, subjects = dt.return_np_raw()
        X_ipf = iterative_polynomial_fitting_2(X, mode = 'old')
        for j in range(3):
            averages.append(np.mean(X_ipf[np.where(y == j)[0]], axis=0) + offset[3*i + j])
            percentiles.append((np.percentile(X_ipf[np.where(y == j)[0]], 5, axis=0) + offset[3*i + j], np.percentile(X_ipf[np.where(y == j)[0]], 95, axis=0) + offset[3*i + j]))
    
    colors = sns.color_palette("Reds", 5)[-1:] * 3 + sns.color_palette("Greens", 5)[-1:] * 3 + sns.color_palette("Blues", 5)[-1:] * 3
    labels = [
              'DEP-Induced (Center) - Normal', 'DEP-Induced (Center) - Lung Cancer', 'DEP-Induced (Center) - Pancreatic Cancer',
              'DEP-Induced (Radial) - Normal', 'DEP-Induced (Radial) - Lung Cancer', 'DEP-Induced (Radial) - Pancreatic Cancer',
              'No Voltage - Normal', 'No Voltage - Lung Cancer', 'No Voltage - Pancreatic Cancer',]
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(7,10))

    for i, (val, (p5, p95), color, label) in enumerate(zip(averages, percentiles, colors, labels)):
        plt.plot(shift, val, color=color)
        plt.fill_between(shift, p5, p95, color=color, alpha=0.2)
        plt.text(shift[0] + 12.5, val[0] - 2000, label, color=color, va='center', fontweight='bold', fontsize=18)

    bar_height = 5000
    bar_x = 575
    bar_y = 74500
    plt.plot([bar_x, bar_x], [bar_y, bar_y + bar_height], color='black', lw=1)
    plt.plot([bar_x - 10, bar_x + 10], [bar_y, bar_y], color='black', lw=1)
    plt.plot([bar_x - 10, bar_x + 10], [bar_y + bar_height, bar_y + bar_height], color='black', lw=1)
    plt.text(bar_x + 10, bar_y + bar_height / 2, '5000', va='center', ha='left', fontsize=18)

    plt.xlim(550, 1800)
    plt.ylim(-3500, 80500)
    from matplotlib.ticker import MultipleLocator
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.set_xlim(550, 1800)
    ax.set_xticks(np.arange(600, 1801, 200))

    ax.xaxis.set_minor_locator(MultipleLocator(100))
    
    plt.setp(plt.gca().get_xticklabels(), fontsize=18)
    plt.yticks([])
    plt.xlabel('Raman Shift (cm$^{-1}$)', fontsize=18)
    plt.ylabel('Intensity (a.u.)', fontsize=18)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.tight_layout()
    plt.savefig(f"fig_5a.png", dpi=300)
    plt.show()