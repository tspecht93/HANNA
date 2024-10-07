import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
# Generell Plot settings
Custom_Blue = (0/255, 95/255, 140/255)  # Converted to [0, 1] range
Custom_Red = (185/255, 40/255, 25/255) # Converted to [0, 1] range

plt.rcParams.update({
        'axes.titlesize': 16,  # Increase the axes title size
        'axes.labelsize': 16,  # Increase the axes labels size
        'axes.linewidth': 2,  # Make the axes box thicker
        'xtick.major.width':2, # Make the x-ticks thicker
        'ytick.major.width': 2, # Make the y-ticks thicker
        'xtick.major.size': 6,  # Make the x-ticks longer
        'ytick.major.size': 6,  # Make the y-ticks longer
        'lines.linewidth': 2,  # Make the error bars thicker
        'lines.markeredgewidth': 5,  # Make the marker edges thicker
        'lines.markersize': 8,  # Increase the marker size (default is 6.0) 
        'font.size' : 16, # Increase the font size
        'legend.frameon': False, # Remove the frame of the legend
        })  

def plot_ln_gamma(x_pred, ln_gammas_pred, temp_real, smiles_1, smiles_2):
    """
    Plots predicted ln(gamma) values for a system at a specific temperature.
    
    Parameters:
    - x_pred: Prediction grid for component fractions.
    - ln_gammas_pred: Predicted ln(gamma) values.
    - temp_real: The temperature in K for the plot title.
    - smiles_1: SMILES string of component 1 for the plot title.
    - smiles_2: SMILES string of component 2 for the plot title.
    """

    # Create the output directory if it doesn't exist
    output_folder = "Output"
    os.makedirs(output_folder, exist_ok=True)

    
    plt.figure(figsize=(6, 6))
    
    plt.plot(x_pred[:, 0], ln_gammas_pred[:, 0], color=Custom_Blue, label='$\ln \gamma_{1}$')
    plt.plot(x_pred[:, 0], ln_gammas_pred[:, 1], color=Custom_Red, label='$\ln \gamma_{2}$')

    plt.xlim(0, 1)
    current_ylim = plt.ylim()
    ylim_diff = current_ylim[1] - current_ylim[0]
    adjusted_higher_ylim = current_ylim[1] + ylim_diff * 0.6  # Upper y-limit is increased by 60% of the difference to have some space for the legend

    plt.ylim(current_ylim[0], adjusted_higher_ylim)

    formatted_temp = "{:.2f}".format(temp_real)  # Format temperature to have only two decimal places

    plt.xlabel('$x_1 / \mathrm{mol}~\mathrm{mol}^{-1}$', fontsize=16)
    plt.ylabel('$\ln \gamma_i$', fontsize=16)
    
    legend_elements = [
        Line2D([0], [0], color=Custom_Blue, lw=2, label='$\ln \gamma_{1}$'),
        Line2D([0], [0], color=Custom_Red, lw=2, label='$\ln \gamma_{2}$')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(f'Predicted activity coefficients at $T$ = {formatted_temp} K', fontsize=16)
    
    # Add SMILES strings above the plot
    plt.figtext(0.5, 0.95, f'SMILES 1: {smiles_1}', ha='center', fontsize=16)
    plt.figtext(0.5, 0.90, f'SMILES 2: {smiles_2}', ha='center', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to accommodate SMILES strings

    # Save the plot as a PNG file in the Output folder
    filename_plot = os.path.join(output_folder, f"System_{smiles_1}_AND_{smiles_2}_AT_{formatted_temp}K.png")
    plt.savefig(filename_plot, dpi=300)
    plt.show()

    # Save data to a CSV file in the Output folder
    data = {
        'SMILES1': [smiles_1] * len(x_pred),
        'SMILES2': [smiles_2] * len(x_pred),
        'T / K': [temp_real] * len(x_pred),
        'x_1 / mol/mol': x_pred[:, 0],
        'ln_gamma_1': ln_gammas_pred[:, 0],
        'ln_gamma_2': ln_gammas_pred[:, 1]
    }
    df = pd.DataFrame(data)
    filename_csv = os.path.join(output_folder, f"System_{smiles_1}_{smiles_2}_{formatted_temp}K.csv")
    df.to_csv(filename_csv, index=False)

    return filename_plot, filename_csv