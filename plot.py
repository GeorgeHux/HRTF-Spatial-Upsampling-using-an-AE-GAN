import matplotlib.pyplot as plt
import numpy as np

def plot_losses(losses, labels, colors, path, filename, title="Loss Curves"):
    """Plot loss curves"""
    params = {
        'axes.labelsize': 10,
        'font.size': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': [6, 4.5]
    }
    plt.rcParams.update(params)
    plt.figure()
    plt.grid(ls='dashed', axis='y', color='0.8')
    for i in range(len(losses)):
        plt.plot(losses[i], label=labels[i], linewidth=2, color=colors[i])
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt_legend = plt.legend()
    frame = plt_legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    plt.savefig(f'{path}/{filename}.png')
    plt.close()


# def plot_hrtf(generated, target, path, filename):
#     x = generated[0, 0, 0, :]
#     y = target[0, 0, 0, :]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     ax1.plot(x)
#     ax1.set_title('recon')
#     ax2.plot(y)
#     ax2.set_title('original')
#     plt.savefig(f"{path}/{filename}.png")
#     plt.close()


def plot_hrtf(generated, target, path, filename):
    # Convert tensors to NumPy arrays
    x = generated[0, 0, 0, :].numpy()
    y = target[0, 0, 0, :].numpy()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    # Plotting the generated data
    ax1.plot(x)
    ax1.set_title('Reconstructed')

    # Plotting the target data
    ax2.plot(y)
    ax2.set_title('Original')

    # Set the Y-axis limits to be the same for both plots
    y_min = min(np.min(x), np.min(y))
    y_max = max(np.max(x), np.max(y))
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    ax1.yaxis.set_tick_params(labelleft=True)
    ax2.yaxis.set_tick_params(labelleft=True)

    # Save the plot to the specified path and filename
    plt.savefig(f"{path}/{filename}.png")
    plt.close()


