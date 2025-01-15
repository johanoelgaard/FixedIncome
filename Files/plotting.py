import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
from scipy.stats import norm, ncx2

DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', 
                  '#d62728', '#2ca02c', 
                  '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', 
                  '#bcbd22', '#17becf'] # matplotlib default color cycle

# set default font size
plt.rcParams.update({'font.size': 12})


def choos_axis(y_min, y_max):
    """Choose an appropriate MultipleLocator step."""
    y_min= (y_min*0.95)//0.1/10
    if y_max <= 0.2:
        y_max = round(y_max+0.005,2)+0.005
    elif y_max <= 0.5:
        y_max = round(y_max+0.05,1)+0.05
    elif y_max <= 1:
        y_max = round(y_max,1)+0.05
    else:
        y_max = round(y_max,1)+0.1
    y_range = y_max - y_min

    if y_range <= 0:
        return 1  # Default step if range is zero or negative
    candidates = [0.005, 0.01, 0.02, 0.05, 0.1]
    for step in candidates:
        num_steps = y_range // step
        if 5 <= num_steps < 10:
            return step, y_min, y_max
    return candidates[1], y_min, y_max # Return step of 0.01 if no suitable step found

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
                  figsize=(10, 6), dpi=300, save_fig=False, show_fig=True):
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
                alpha=data.get('alpha', 0.75),
                marker=data.get('marker', '.'),
                s=data.get('s', 10),
                **data.get('kwargs', {})
            )
        handles.append(plot_)
    
    # Adjust y-limits for left axis considering only non-NaN values
    if left_data:
        all_y_left = np.concatenate([np.array(data['y'])[~np.isnan(data['y'])] for data in left_data])
        y_min_left, y_max_left = all_y_left.min(), all_y_left.max()
        step_left, y_min_left, y_max_left = choos_axis(y_min_left, y_max_left)
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
                    alpha=data.get('alpha', 0.75),
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
        lines = []
        for key, value in text.items():
            if isinstance(value, (list, np.ndarray)):
                    # Format each element in the list/array
                    value_str = ", ".join([f"{v:.5f}" for v in value])
            elif isinstance(value, (float, np.float64, np.float32)):
                    # Format single numerical value
                    value_str = f"{value:.5f}"
            else:
                value_str = value
            lines.append(f"{key} = {value_str}")
        textstr = "\n".join(lines)

        # textstr = "\n".join([f"{key} = {value}" for key, value in text.items()])
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
    
    if show_fig:
        plt.show()
    
    plt.close()

def histogram(data, bins=100, title="", text=None, theoretical=None, theoretical_params=None,
              xlabel="Rate", ylabel="Frequency",
              legend_loc='upper right', text_loc='upper left',
              figsize=(10, 6), dpi=300, save_fig=False, show_fig=True):
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
        text_coord = (0.015, 0.975)
        vertalign = 'top'
        horalign = 'left'
    elif text_loc == 'upper right':
        text_coord = (0.985, 0.975)
        vertalign = 'top'
        horalign = 'right'
    elif text_loc == 'lower left':
        text_coord = (0.015, 0.025)
        vertalign = 'bottom'
        horalign = 'left'
    elif text_loc == 'lower right':
        text_coord = (0.985, 0.025)
        vertalign = 'bottom'
        horalign = 'right'
    if text:
        lines = []
        for key, value in text.items():
            if isinstance(value, (list, np.ndarray)):
                    # Format each element in the list/array
                    value_str = ", ".join([f"{v:.5f}" for v in value])
            elif isinstance(value, (float, np.float64, np.float32)):
                    # Format single numerical value
                    value_str = f"{value:.5f}"
            else:
                value_str = value
            lines.append(f"{key} = {value_str}")
        textstr = "\n".join(lines)
        props = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
        ax.text(
            text_coord[0], text_coord[1], textstr, transform=ax.transAxes,
            verticalalignment=vertalign,horizontalalignment=horalign,bbox=props
        )
    
    # Combine legends
    ax.legend(loc=legend_loc)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_fig)
        print(f"Figure saved as {save_fig}")
    
    if show_fig:
        plt.show()
    
    plt.close()

def fit(data, residual, title="", text=None, text_res=None, 
        ylabel="Rates", xlabel="Time to Maturity", ylabel_residual="Residuals",
        legend_loc='upper right', text_loc='lower right', text_res_loc='lower right',
        figsize=(10, 6), dpi=300, save_fig=False, show_fig=True):
    """
    Plots fitted functions alongside original data and displays residuals in a separate subplot.

    Parameters:
    - data (list of dict): Each dict should have keys 'label', 'x', 'y', and optionally 'color', 'alpha', 'marker', 'kwargs'.
      Example: [{'label': '$f^*$', 'x': T, 'y': f_star, 'color': 'blue'}, {'label': '$\hat{f}$', 'x': T, 'y': f_hat, 'color': 'red'}]
    - residual (list of dict): Similar structure to data for residuals.
      Example: [{'label': 'Residuals', 'x': T, 'y': residual, 'color': 'black'}]
    - text (dict, optional): Dictionary containing text to display on the first subplot.
      Example: {'$r_0$': r0_hat, '$a$': a_hat, ...}
    - text_res (dict, optional): Dictionary containing text to display on the residuals subplot.
      Example: {'MSE': mse}
    - title (str, optional): Title of the plot.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis of the first subplot.
    - ylabel_residual (str, optional): Label for the y-axis of the residuals subplot.
    - legend_loc (str, optional): Location of the legend in the first subplot.
    - text_loc (str, optional): Location of the text box in the first subplot ('upper left', 'upper right', etc.).
    - text_res_loc (str, optional): Location of the text box in the residuals subplot.
    - figsize (tuple, optional): Figure size.
    - dpi (int, optional): Dots per inch for the figure.
    - save_fig (str or False, optional): Filename to save the figure. If False, the figure is not saved.
    - show_fig (bool, optional): Whether to display the figure.
    """
    
    # Create figure and two subplots (axes)
    fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
    
    # ------------------ First Subplot: Data and Fit ------------------ #
    color_idx = 0
    handles = []

    for plot_data in data:
        label = plot_data.get('label', f'Data {color_idx}')
        x = plot_data['x']
        y = plot_data['y']
        
        if 'color' in plot_data.keys():
            color = plot_data['color']
        else:
            color=DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
            color_idx += 1
        
        alpha = plot_data.get('alpha', 0.75)
        marker = plot_data.get('marker', '.')
        s = plot_data.get('s', 20)
        linestyle = plot_data.get('linestyle', None)  # For line plots if needed
        linewidth = plot_data.get('linewidth', 1.5)
        kwargs = plot_data.get('kwargs', {})
        
        if plot_data.get('type') == 'line':
            plot = ax[0].plot(x, y, label=label, color=color,
                             alpha=alpha, linestyle=linestyle,
                             linewidth=linewidth, **kwargs)
            handles.extend(plot)
        else:
            plot = ax[0].scatter(x, y, label=label, color=color,
                                 alpha=alpha, marker=marker, s=s, **kwargs)
            handles.append(plot)
    
    # Set y-axis locator for the first subplot
    if data:
        all_y = np.concatenate([np.array(d['y'])[~np.isnan(d['y'])] for d in data])
        y_min, y_max = all_y.min(), all_y.max()
        step, y_min, y_max = choos_axis(y_min, y_max)
        ax[0].set_ylim([y_min, y_max])
        ax[0].yaxis.set_major_locator(MultipleLocator(step))
        # Add horizontal grid lines
        for tick in ax[0].get_yticks():
            ax[0].axhline(y=tick, color='gray', linestyle='--', linewidth=1)
    
    ax[0].set_ylabel(ylabel)
    
    # Combine legends
    if handles:
        ax[0].legend(handles=handles, loc=legend_loc)
    # Add text box to the first subplot if provided
    if text:
        if text_loc == 'upper left':
            text_coord = (0.015, 0.95)
            vertalign = 'top'
            horalign = 'left'
        elif text_loc == 'upper right':
            text_coord = (0.985, 0.95)
            vertalign = 'top'
            horalign = 'right'
        elif text_loc == 'lower left':
            text_coord = (0.015, 0.05)
            vertalign = 'bottom'
            horalign = 'left'
        elif text_loc == 'lower right':
            text_coord = (0.985, 0.05)
            vertalign = 'bottom'
            horalign = 'right'
        lines = []
        for key, value in text.items():
            if isinstance(value, (list, np.ndarray)):
                    # Format each element in the list/array
                    value_str = ", ".join([f"{v:.5f}" for v in value])
            elif isinstance(value, (float, np.float64, np.float32)):
                    # Format single numerical value
                    value_str = f"{value:.5f}"
            else:
                value_str = value
            lines.append(f"{key} = {value_str}")
        textstr = "\n".join(lines)

        props = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='grey')
        ax[0].text(
            text_coord[0], text_coord[1], textstr, transform=ax[0].transAxes,
            verticalalignment=vertalign, horizontalalignment=horalign,
            bbox=props
        )
    
    # ------------------ Second Subplot: Residuals ------------------ #
    residual_handles = []
    
    for res_data in residual:
        label = res_data.get('label', f'Residual {color_idx}')
        x = res_data['x']
        y = res_data['y']
        if 'color' in res_data.keys() and res_data['color'] == 'cycle':
            color=DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
            color_idx += 1        
        elif 'color' in res_data.keys():
            color = res_data['color']
        else:
            color = 'black'
        alpha = res_data.get('alpha', 0.75)
        marker = res_data.get('marker', '.')
        s = res_data.get('s', 20)
        kwargs = res_data.get('kwargs', {})
        
        plot = ax[1].scatter(x, y, label=label, color=color,
                             alpha=alpha, marker=marker, s=s, **kwargs)
        residual_handles.append(plot)
    
    # Set y-axis locator for the residuals subplot
    if residual:
        all_res = np.concatenate([np.array(r['y'])[~np.isnan(r['y'])] for r in residual])
        res_min, res_max = all_res.min(), all_res.max()
        res_range = res_max - res_min
        
        # Determine a reasonable base for MultipleLocator
        if res_range == 0:
            base_res = 1e-6  # Arbitrary small base
        else:
            exponent = int(np.floor(np.log10(res_range))) if res_range > 0 else 0
            if res_range // (10 ** (exponent-1)) < 10:
                base_res = 10 ** (exponent-1)
            elif res_range // (2*10 ** (exponent-1)) < 10:
                base_res = 2*10 ** (exponent-1)
            elif res_range // (5*10 ** (exponent-1)) < 10:
                base_res = 5*10 ** (exponent-1)
            else:
                base_res = 10 ** exponent

        ax[1].yaxis.set_major_locator(MultipleLocator(base_res))
        # Add horizontal grid lines
        for tick in ax[1].get_yticks():
            ax[1].axhline(y=tick, color='gray', linestyle='--', linewidth=1)
    
    ax[1].set_ylabel(ylabel_residual)
    
    # Combine legends for residuals
    if residual_handles:
        ax[1].legend(handles=residual_handles, loc=legend_loc)
    
    # Add text box to the residuals subplot if provided
    if text_res:
        if text_loc == 'upper left':
            text_coord = (0.015, 0.95)
            vertalign = 'top'
            horalign = 'left'
        elif text_loc == 'upper right':
            text_coord = (0.985, 0.95)
            vertalign = 'top'
            horalign = 'right'
        elif text_loc == 'lower left':
            text_coord = (0.015, 0.05)
            vertalign = 'bottom'
            horalign = 'left'
        elif text_loc == 'lower right':
            text_coord = (0.985, 0.05)
            vertalign = 'bottom'
            horalign = 'right'
        lines_res = []
        for key, value in text.items():
            if isinstance(value, (list, np.ndarray)):
                    # Format each element in the list/array
                    value_str = ", ".join([f"{v:.5f}" for v in value])
            elif isinstance(value, (float, np.float64, np.float32)):
                    # Format single numerical value
                    value_str = f"{value:.5f}"
            else:
                value_str = value
            lines_res.append(f"{key} = {value_str}")
        textstr_res = "\n".join(lines_res)
        props_res = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='grey')
        ax[1].text(
            text_coord[0], text_coord[1], textstr_res, transform=ax[1].transAxes,
            verticalalignment=vertalign, horizontalalignment=horalign,
            bbox=props_res
        )
    
    # Set common x-label
    ax[1].set_xlabel(xlabel)
    
    # Set the title for the entire figure
    if title:
        ax[0].set_title(title)
    
    # Adjust layout
    plt.tight_layout()  # Leave space for suptitle
    
    # Handle saving, showing, and returning the figure
    if save_fig:
        plt.savefig(save_fig)
        print(f"Figure saved as {save_fig}")
    
    if show_fig:
        plt.show()
    
    plt.close()
