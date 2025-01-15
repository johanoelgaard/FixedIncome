import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.stats import norm, ncx2

DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', 
                  '#d62728', '#2ca02c', 
                  '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', 
                  '#bcbd22', '#17becf'] # matplotlib default color cycle

# set default font size
plt.rcParams.update({'font.size': 12})


def choose_multiple(y_min, y_max):
    """Choose an appropriate MultipleLocator step."""
    y_min= (y_min*0.95)//0.1/10
    if y_max < 0.2:
        y_max = round(y_max+0.005,2)+0.005
    elif y_max < 0.5:
        y_max = round(y_max+0.05,1)+0.05
    else:
        y_max = round(y_max+0.1,1)+0.1
    y_range = y_max - y_min

    if y_range <= 0:
        return 1  # Default step if range is zero or negative
    candidates = [0.01, 0.02, 0.05, 0.1]
    for step in candidates:
        num_steps = y_range // step
        if 5 <= num_steps <= 10:
            return step, y_min, y_max
    return candidates[-1], y_min, y_max # Return the largest step if no suitable step found

def choose_ticks(y_min, y_max, ticks):
    """Choose an appropriate MultipleLocator step dependent on ticks in first axis."""
    y_min= 0
    y_max = (y_max*1.05)//0.1/10+0.1

    step = y_max / ticks
    if y_max < 0.2:
        step = round(step, 2)
    else:
        step = round(step, 1)

    return y_min, y_max, step

def rates(left_data, right_data=None, title="", text=None,
                  xlabel="Time to Maturity", ylabel="Rates", ylabel_right="Bond Prices",
                  legend_loc='lower right', text_loc='lower left',
                  figsize=(10, 6), dpi=300, save_fig=False, show_fig=True, return_fig=False):
    """
    Plots data on one or two y-axes.

    Parameters:
    - left_data (list of dict): Each dict should have keys 'label', 'x', 'y', and optionally 'color', 'alpha', 'marker', 'kwargs'.
      Example: {'label':'Spot Price', 'x':T, 'y':spot_price, 'color':'blue'}
    - right_data (list of dict, optional): Similar structure to left_data for the right y-axis.
    - text (dict, optional): Dictionary containing text to display on the plot.
      Example: {'$\\beta$': beta, '$r_0$': r0}
    - title (str, optional): Title of the plot.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the left y-axis.
    - ylabel_right (str, optional): Label for the right y-axis.
    - figsize (tuple, optional): Figure size.
    - dpi (int, optional): Dots per inch for the figure.
    """

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)


    if not right_data:
        axis_ =""
    else:
        axis_ = "(left)"

    color_idx = 0
    # Plot data on the left axis
    handles = []
    for data in left_data:
        if 'color' in data.keys():
            color=data['color']
        else:
            color=DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
            color_idx += 1
        
        if 'type' in data.keys() and data['type'] == 'line':
            plot_ = ax1.plot(
                data['x'], 
                data['y'], 
                label=f"{data['label']} {axis_}",
                color=color, 
                alpha=data.get('alpha', 1),
                linestyle=data.get('linestyle', '-'),
                linewidth=data.get('linewidth', 1.5),
                **data.get('kwargs', {})
            )
            
        else:
            plot_ = ax1.scatter(
                data['x'], 
                data['y'], 
                label=f"{data['label']} {axis_}",
                color=color, 
                alpha=data.get('alpha', 0.5),
                marker=data.get('marker', '.'),
                s=data.get('s', 10),
                **data.get('kwargs', {})
            )
        handles.append(plot_)
    
    # Adjust y-limits for left axis considering only non-NaN values
    if left_data:
        all_y_left = np.concatenate([np.array(data['y'])[~np.isnan(data['y'])] for data in left_data])
        y_min_left, y_max_left = all_y_left.min(), all_y_left.max()
        step_left, y_min_left, y_max_left = choose_multiple(y_min_left, y_max_left)
        ax1.set_ylim([y_min_left, y_max_left])
        ticks = int((y_max_left-y_min_left)//step_left)
        ax1.yaxis.set_major_locator(MultipleLocator(step_left))
        # Add horizontal grid lines
        for tick in ax1.get_yticks():
            ax1.axhline(y=tick, color='gray', linestyle='--', linewidth=1)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    # Plot data on the right axis if provided
    if right_data:
        ax2 = ax1.twinx()
        for data in right_data:
            if 'color' in data.keys():
                color=data['color']
            else:
                color=DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
                color_idx += 1
            if 'type' in data.keys() and data['type'] == 'line':
                plot_ = ax2.plot(
                    data['x'], 
                    data['y'], 
                    label=f"{data['label']} {axis_}",
                    color=color, 
                    alpha=data.get('alpha', 1),
                    linestyle=data.get('linestyle', '-'),
                    linewidth=data.get('linewidth', 1.5),
                    **data.get('kwargs', {})
                )
                
            else:
                plot_ = ax2.scatter(
                    data['x'], 
                    data['y'], 
                    label=f"{data['label']} {axis_}",
                    color=color, 
                    alpha=data.get('alpha', 0.5),
                    marker=data.get('marker', '.'),
                    s=data.get('s', 10),
                    **data.get('kwargs', {})
                )
            handles.append(plot_)
        # Adjust y-limits for right axis considering only non-NaN values
        all_y_right = np.concatenate([np.array(data['y'])[~np.isnan(data['y'])] for data in right_data])
        y_min_right, y_max_right = all_y_right.min(), all_y_right.max()
        y_min_right, y_max_right, step_right = choose_ticks(y_min_right, y_max_right, ticks)
        # updating y_max_right
        y_max_right = (y_max_left-y_min_right)/step_left*step_right

        ax2.set_ylim([y_min_right, y_max_right])
        ax2.set_ylabel(ylabel_right)

        ax2.yaxis.set_major_locator(MultipleLocator(step_right))
    else:
        ax2 = None

    # Combine legends from both axes
    if ax2:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
    else:
        handles, labels = ax1.get_legend_handles_labels()
    
    if handles and labels:
        ax1.legend(handles, labels, loc=legend_loc)

    # Add text box if provided
    if text_loc == 'upper left':
        text_loc = (0.015, 0.975)
        vertalign = 'top'
        horalign = 'left'
    elif text_loc == 'upper right':
        text_loc = (0.985, 0.975)
        vertalign = 'top'
        horalign = 'right'
    elif text_loc == 'lower left':
        text_loc = (0.015, 0.025)
        vertalign = 'bottom'
        horalign = 'left'
    elif text_loc == 'lower right':
        text_loc = (0.985, 0.025)
        vertalign = 'bottom'
        horalign = 'right'
    if text:
        textstr = "\n".join([f"{key} = {value}" for key, value in text.items()])
        props = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
        ax1.text(
            text_loc[0], text_loc[1], textstr, transform=ax1.transAxes,
            verticalalignment=vertalign,horizontalalignment=horalign,bbox=props
        )

    # Set the title
    if title:
        plt.title(title)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig)
        print(f"Figure saved as {save_fig}")

    if return_fig:
        return fig
    
    if show_fig:
        plt.show()
    
    plt.close()


def histogram(data, bins=100, theoretical=None, theoretical_params=None,
              title="", text=None, xlabel="Rate", ylabel="Frequency",
              legend_loc='upper right', text_loc='upper left',
              figsize=(10, 6), dpi=300, save_fig=False, show_fig=True, return_fig=False):
    """
    Plots a histogram of the data with an optional theoretical distribution overlay.

    Parameters:
    - data (array-like): The data to be histogrammed.
    - bins (int or sequence): Number of histogram bins or bin edges.
    - theoretical (str, optional): Type of theoretical distribution to overlay ('chi2', 'normal', etc.).
    - theoretical_params (dict, optional): Parameters for the theoretical distribution.
      For 'chi2', expected keys are 'df' and 'nc'.
      For 'normal', expected keys are 'mu' and 'std'.
    - title (str, optional): Title of the plot.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.
    - text (dict, optional): Dictionary containing text to display on the plot.
      Example: {'$r_0$': r0, '$a$': a, ...}
    - legend_loc (str, optional): Location of the legend.
    - text_loc (str, optional): Location of the text box ('upper left', 'upper right', etc.).
    - figsize (tuple, optional): Figure size.
    - dpi (int, optional): Dots per inch for the figure.
    - save_fig (str or False, optional): Filename to save the figure. If False, the figure is not saved.
    - show_fig (bool, optional): Whether to display the figure.
    - return_fig (bool, optional): Whether to return the figure object.
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot histogram
    hist_data = ax.hist(data, bins=bins, color='blue', edgecolor='black', label='Simulated', density=True, alpha=0.75)
    
    # Plot theoretical distribution if specified
    if theoretical and theoretical_params:
        bin_edges = hist_data[1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        if theoretical.lower() == 'chi2':
            df = theoretical_params.get('df')
            nc = theoretical_params.get('nc', 0)
            if df is None:
                raise ValueError("For 'chi2' distribution, 'df' parameter must be provided in theoretical_params.")
            pdf_values = ncx2.pdf(bin_centers / theoretical_params.get('scale', 1), df=df, nc=nc) / theoretical_params.get('scale', 1)
            label_theoretical = "Theoretical (Noncentral Chi2)"
            color_theoretical = 'red'
        elif theoretical.lower() == 'normal':
            mu = theoretical_params.get('mu')
            std = theoretical_params.get('std')
            if mu is None or std is None:
                raise ValueError("For 'normal' distribution, 'mu' and 'std' parameters must be provided in theoretical_params.")
            pdf_values = norm.pdf(bin_centers, loc=mu, scale=std)
            label_theoretical = "Theoretical (Normal)"
            color_theoretical = 'red'
        else:
            raise ValueError(f"Theoretical distribution '{theoretical}' is not supported.")
        
        # Overlay theoretical distribution
        ax.scatter(bin_centers, pdf_values, color=color_theoretical, label=label_theoretical, s=5)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add text box if provided
    if text_loc == 'upper left':
        text_loc = (0.015, 0.975)
        vertalign = 'top'
        horalign = 'left'
    elif text_loc == 'upper right':
        text_loc = (0.985, 0.975)
        vertalign = 'top'
        horalign = 'right'
    elif text_loc == 'lower left':
        text_loc = (0.015, 0.025)
        vertalign = 'bottom'
        horalign = 'left'
    elif text_loc == 'lower right':
        text_loc = (0.985, 0.025)
        vertalign = 'bottom'
        horalign = 'right'
    if text:
        textstr = "\n".join([f"{key} = {value}" for key, value in text.items()])
        props = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
        ax.text(
            text_loc[0], text_loc[1], textstr, transform=ax.transAxes,
            verticalalignment=vertalign,horizontalalignment=horalign,bbox=props
        )
    
    # Combine legends
    ax.legend(loc=legend_loc)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_fig)
        print(f"Figure saved as {save_fig}")
    
    if return_fig:
        return fig
    
    if show_fig:
        plt.show()
    
    plt.close()
