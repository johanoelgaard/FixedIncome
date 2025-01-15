import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Define a default color cycle
DEFAULT_COLORS = [
    "blue", "red", "green", "black", 
    "orange", "purple", "brown", "pink"
]

def _choose_tick_interval(range_val, max_ticks=6):
    """
    Choose a tick interval from multiples of 0.01, 0.02, 0.05, 0.1, etc.,
    to have at most max_ticks on the y-axis.

    Parameters
    ----------
    range_val : float
        The range of y-values (ymax - ymin).
    max_ticks : int
        Maximum number of ticks desired.

    Returns
    -------
    tick_interval : float
        The chosen interval between ticks.
    """
    if range_val <= 0:
        return 0.01  # Default small interval for non-positive ranges

    exponent = np.floor(np.log10(range_val))
    fraction = range_val / (10**exponent)

    if fraction <= 1:
        tick_interval = 0.1 * (10**exponent)
    elif fraction <= 2:
        tick_interval = 0.2 * (10**exponent)
    elif fraction <= 5:
        tick_interval = 0.5 * (10**exponent)
    else:
        tick_interval = 1.0 * (10**exponent)

    return tick_interval

def _auto_limits_and_ticks(values, max_ticks=6):
    """
    Compute y-limits with padding and determine tick positions as multiples.

    Parameters
    ----------
    values : array-like
        1D array of y-values.
    max_ticks : int
        Maximum number of ticks desired.

    Returns
    -------
    (ymin, ymax) : tuple
        The y-axis limits.
    yticks : array-like
        Positions of the y-axis ticks.
    """
    val_min = np.min(values)
    val_max = np.max(values)

    # Handle cases where min and max are very close
    if np.isclose(val_min, val_max):
        val_min -= 0.1 * abs(val_min) if val_min != 0 else 0.001
        val_max += 0.1 * abs(val_max) if val_max != 0 else 0.001

    # Add padding
    pad = 0.05 * (val_max - val_min)
    ymin = val_min - pad
    ymax = val_max + pad

    range_val = ymax - ymin
    tick_interval = _choose_tick_interval(range_val, max_ticks=max_ticks)

    # Round ymin and ymax to the nearest multiple of tick_interval
    ymin_rounded = np.floor(ymin / tick_interval) * tick_interval
    ymax_rounded = np.ceil(ymax / tick_interval) * tick_interval

    # Generate tick positions
    yticks = np.arange(ymin_rounded, ymax_rounded + tick_interval, tick_interval)

    return (ymin_rounded, ymax_rounded), yticks

# Define a default color cycle
DEFAULT_COLORS = [
    "blue", "red", "green", "black", 
    "orange", "purple", "brown", "pink"
]

def plot_single_axis_scatter(
    data_list, 
    x_key="x", 
    y_key="y", 
    title="",
    xlabel="x",
    ylabel="y",
    n_ticks=6,
    colors=None,
    figsize=(10,6),
    dpi=300
):
    """
    Plots multiple scatter datasets on a single axis.
    Automatically adjusts y-axis limits and tick intervals.

    Parameters
    ----------
    data_list : list of dict
        Each dict should contain at least {x_key: <array>, y_key: <array>, "name": <str>}.
        Optional keys: "color", "alpha", "marker".
    x_key : str, optional
        Key for x-data in each dict. Default is "x".
    y_key : str, optional
        Key for y-data in each dict. Default is "y".
    title : str, optional
        Plot title. Default is "".
    xlabel : str, optional
        Label for the x-axis. Default is "x".
    ylabel : str, optional
        Label for the y-axis. Default is "y".
    n_ticks : int, optional
        Number of ticks on the y-axis. Default is 6.
    colors : list or None, optional
        List of colors to cycle through. If None, uses DEFAULT_COLORS.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    dpi : int, optional
        Dots per inch for the figure. Default is 300.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axis objects.
    """
    if colors is None:
        color_cycle = DEFAULT_COLORS
    else:
        color_cycle = colors

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Gather all y-values to determine y-axis limits and ticks
    all_ys = np.concatenate([d[y_key] for d in data_list])
    (ymin, ymax), yticks = _auto_limits_and_ticks(all_ys, max_ticks=n_ticks)
    ax.set_ylim([ymin, ymax])
    ax.set_yticks(yticks)

    # Plot each dataset
    legend_handles = []
    for i, d in enumerate(data_list):
        plot_color  = d.get("color",  color_cycle[i % len(color_cycle)])
        plot_alpha  = d.get("alpha", 1)
        plot_marker = d.get("marker", ".")

        scatter = ax.scatter(
            d[x_key],
            d[y_key],
            label=d.get("name", f"Series {i+1}"),
            alpha=plot_alpha,
            marker=plot_marker,
            color=plot_color
        )
        legend_handles.append(scatter)

    # Draw horizontal grid lines
    for tick in yticks:
        ax.axhline(y=tick, color="gray", linestyle="--", linewidth=1, zorder=0)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add legend
    ax.legend(handles=legend_handles, loc="best")

    plt.tight_layout()
    return fig, ax

def plot_dual_axis_scatter(
    left_data_list,
    right_data_list,
    x_key="x", 
    y_key="y", 
    title="",
    xlabel="x",
    left_ylabel="Left Axis",
    right_ylabel="Right Axis",
    n_ticks=6,
    colors=None,
    figsize=(10,6),
    dpi=300
):
    """
    Plots multiple scatter datasets on two different y-axes (left and right).
    Ensures both axes share the same y-axis tick intervals for alignment.

    Parameters
    ----------
    left_data_list : list of dict
        Data to plot on the left y-axis. Each dict should contain {x_key: <array>, y_key: <array>, "name": <str>}.
        Optional keys: "color", "alpha", "marker".
    right_data_list : list of dict
        Data to plot on the right y-axis. Same structure as left_data_list.
    x_key : str, optional
        Key for x-data in each dict. Default is "x".
    y_key : str, optional
        Key for y-data in each dict. Default is "y".
    title : str, optional
        Plot title. Default is "".
    xlabel : str, optional
        Label for the x-axis. Default is "x".
    left_ylabel : str, optional
        Label for the left y-axis. Default is "Left Axis".
    right_ylabel : str, optional
        Label for the right y-axis. Default is "Right Axis".
    n_ticks : int, optional
        Number of ticks on both y-axes. Default is 6.
    colors : list or None, optional
        List of colors to cycle through. If None, uses DEFAULT_COLORS.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    dpi : int, optional
        Dots per inch for the figure. Default is 300.

    Returns
    -------
    fig, (ax_left, ax_right) : matplotlib.figure.Figure, tuple of matplotlib.axes.Axes
        The created figure and both axis objects.
    """
    if colors is None:
        color_cycle = DEFAULT_COLORS
    else:
        color_cycle = colors

    fig, ax_left = plt.subplots(figsize=figsize, dpi=dpi)
    ax_right = ax_left.twinx()

    # Gather all y-values from both axes to determine common tick interval
    all_left_ys = np.concatenate([d[y_key] for d in left_data_list])
    all_right_ys = np.concatenate([d[y_key] for d in right_data_list])
    combined_ys = np.concatenate([all_left_ys, all_right_ys])

    (ymin, ymax), _ = _auto_limits_and_ticks(combined_ys, max_ticks=n_ticks)
    range_val = ymax - ymin
    tick_interval = _choose_tick_interval(range_val, max_ticks=n_ticks)

    # Set both y-axes with the same limits and tick intervals
    ymin_rounded = np.floor(ymin / tick_interval) * tick_interval
    ymax_rounded = np.ceil(ymax / tick_interval) * tick_interval
    yticks_common = np.arange(ymin_rounded, ymax_rounded + tick_interval, tick_interval)

    ax_left.set_ylim([ymin_rounded, ymax_rounded])
    ax_right.set_ylim([ymin_rounded, ymax_rounded])

    ax_left.set_yticks(yticks_common)
    ax_right.set_yticks(yticks_common)

    # Draw horizontal grid lines
    for tick in yticks_common:
        ax_left.axhline(y=tick, color="gray", linestyle="--", linewidth=1, zorder=0)

    # Plot left data
    left_handles = []
    for i, d in enumerate(left_data_list):
        plot_color  = d.get("color",  color_cycle[i % len(color_cycle)])
        plot_alpha  = d.get("alpha", 1)
        plot_marker = d.get("marker", ".")

        scatter = ax_left.scatter(
            d[x_key],
            d[y_key],
            label=d.get("name", f"Left {i+1}"),
            alpha=plot_alpha,
            marker=plot_marker,
            color=plot_color
        )
        left_handles.append(scatter)

    # Plot right data
    right_handles = []
    for i, d in enumerate(right_data_list):
        idx = i + len(left_data_list)  # Continue color cycle
        plot_color  = d.get("color",  color_cycle[idx % len(color_cycle)])
        plot_alpha  = d.get("alpha", 1)
        plot_marker = d.get("marker", ".")

        scatter = ax_right.scatter(
            d[x_key],
            d[y_key],
            label=d.get("name", f"Right {i+1}"),
            alpha=plot_alpha,
            marker=plot_marker,
            color=plot_color
        )
        right_handles.append(scatter)

    # Set labels and title
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(left_ylabel)
    ax_right.set_ylabel(right_ylabel)
    ax_left.set_title(title)

    # Combine legends
    handles = left_handles + right_handles
    labels = [h.get_label() for h in handles]
    ax_left.legend(handles, labels, loc="best")

    plt.tight_layout()
    return fig, (ax_left, ax_right)

def plot_histogram(
    data,
    bins=50,
    title="",
    xlabel="Value",
    ylabel="Frequency",
    color=None,
    edgecolor="black",
    alpha=1,
    figsize=(10, 6),
    dpi=300,
    density=False,
    overlay_pdf=None,
    overlay_pdf_label="PDF"
):
    """
    Plots a histogram of the provided data with an optional PDF overlay.
    Ensures y-axis ticks are multiples of standard intervals.

    Parameters
    ----------
    data : array-like
        1D data to histogram.
    bins : int, optional
        Number of histogram bins. Default is 50.
    title : str, optional
        Plot title. Default is "".
    xlabel : str, optional
        Label for the x-axis. Default is "Value".
    ylabel : str, optional
        Label for the y-axis. Default is "Frequency".
    color : str or None, optional
        Color of the histogram bars. If None, defaults to "blue".
    edgecolor : str, optional
        Color of the histogram bar edges. Default is "black".
    alpha : float, optional
        Transparency of the histogram bars. Default is 1.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    dpi : int, optional
        Dots per inch for the figure. Default is 300.
    density : bool, optional
        If True, normalize the histogram. Default is False.
    overlay_pdf : dict or None, optional
        If provided, should contain {"x": array, "y": array} for the PDF to overlay.
    overlay_pdf_label : str, optional
        Label for the overlaid PDF in the legend. Default is "PDF".

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axis objects.
    """
    if color is None:
        color = "blue"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    counts, edges, _ = ax.hist(
        data, bins=bins, color=color, edgecolor=edgecolor,
        alpha=alpha, density=density, label="Data"
    )

    # Determine y-axis tick interval
    y_min, y_max = ax.get_ylim()
    tick_interval = _choose_tick_interval(y_max - y_min, max_ticks=6)
    yticks = np.arange(
        np.floor(y_min / tick_interval) * tick_interval,
        np.ceil(y_max / tick_interval) * tick_interval + tick_interval,
        tick_interval
    )
    ax.set_yticks(yticks)

    # Draw horizontal grid lines
    for tick in yticks:
        ax.axhline(y=tick, color="gray", linestyle="--", linewidth=1, zorder=0)

    # Overlay PDF if provided
    if overlay_pdf is not None:
        xpdf = overlay_pdf["x"]
        ypdf = overlay_pdf["y"]
        ax.scatter(xpdf, ypdf, color="red", s=5, label=overlay_pdf_label)

    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add legend
    ax.legend(loc="best")

    plt.tight_layout()
    return fig, ax

def plot_confidence_bands(
    x, 
    y_lower, 
    y_upper,
    y_mean=None,
    title="",
    xlabel="x",
    ylabel="y",
    n_ticks=6,
    figsize=(10,6),
    dpi=300,
    color_lower="red",
    color_upper="red",
    color_mean="blue",
    alpha=1,
    marker="."
):
    """
    Plots lower and upper confidence bands with an optional mean line.
    Ensures y-axis ticks are multiples of standard intervals.

    Parameters
    ----------
    x : array-like
        x-axis values (e.g., time to maturity).
    y_lower : array-like
        Lower confidence band values.
    y_upper : array-like
        Upper confidence band values.
    y_mean : array-like or None, optional
        Mean or central trend line. If None, only bands are plotted. Default is None.
    title : str, optional
        Plot title. Default is "".
    xlabel : str, optional
        Label for the x-axis. Default is "x".
    ylabel : str, optional
        Label for the y-axis. Default is "y".
    n_ticks : int, optional
        Number of ticks on the y-axis. Default is 6.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    dpi : int, optional
        Dots per inch for the figure. Default is 300.
    color_lower : str, optional
        Color for the lower confidence band. Default is "red".
    color_upper : str, optional
        Color for the upper confidence band. Default is "red".
    color_mean : str, optional
        Color for the mean line. Default is "blue".
    alpha : float, optional
        Transparency for scatter points. Default is 1.
    marker : str, optional
        Marker style for scatter points. Default is ".".

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Combine all y-values to determine y-axis limits and ticks
    combined_ys = np.concatenate([y_lower, y_upper] + ([y_mean] if y_mean is not None else []))
    (ymin, ymax), yticks = _auto_limits_and_ticks(combined_ys, max_ticks=n_ticks)
    ax.set_ylim([ymin, ymax])
    ax.set_yticks(yticks)

    # Plot lower confidence band
    plb = ax.scatter(x, y_lower, label="Lower bound", 
                     color=color_lower, marker=marker, alpha=alpha)
    # Plot upper confidence band
    pub = ax.scatter(x, y_upper, label="Upper bound", 
                     color=color_upper, marker=marker, alpha=alpha)

    # Optionally plot mean
    if y_mean is not None:
        pm = ax.scatter(x, y_mean, label="Mean", 
                        color=color_mean, marker=marker, alpha=alpha)

    # Draw horizontal grid lines
    for tick in yticks:
        ax.axhline(y=tick, color="gray", linestyle="--", linewidth=1, zorder=0)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add legend
    if y_mean is not None:
        ax.legend(loc="best")
    else:
        ax.legend([plb, pub], ["Lower bound", "Upper bound"], loc="best")

    plt.tight_layout()
    return fig, ax


def plot_simulation(
    data_list, 
    title="",
    xlabel="x",
    ylabel="y",
    n_ticks=6,
    colors=None,
    figsize=(10,6),
    dpi=300
):
    """
    Plots multiple scatter datasets on a single or dual axes.
    This function is versatile enough to handle standard scatter plots,
    simulations with multiple paths, and confidence bands.

    Parameters
    ----------
    data_list : list of dict
        Each dict should contain:
            - 'x': array-like, x-data
            - 'y': array-like, y-data
            - 'name': str, label for the dataset
        Optional keys for styling:
            - 'color': str, color of the scatter points
            - 'alpha': float, transparency level
            - 'marker': str, marker style
            - 'axis': str, 'left' or 'right' (for dual-axis plots)
    title : str, optional
        Plot title. Default is "".
    xlabel : str, optional
        Label for the x-axis. Default is "x".
    ylabel : str, optional
        Label for the y-axis. Default is "y".
    n_ticks : int, optional
        Number of ticks on the y-axis. Default is 6.
    colors : list or None, optional
        List of colors to cycle through. If None, uses DEFAULT_COLORS.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    dpi : int, optional
        Dots per inch for the figure. Default is 300.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes or tuple
        The created figure and axis objects. Returns a tuple (fig, ax_left, ax_right)
        if dual axes are used.
    """
    if colors is None:
        color_cycle = DEFAULT_COLORS
    else:
        color_cycle = colors

    # Determine if dual-axis is needed
    dual_axis = any(d.get("axis", "left") == "right" for d in data_list)

    if dual_axis:
        fig, ax_left = plt.subplots(figsize=figsize, dpi=dpi)
        ax_right = ax_left.twinx()
    else:
        fig, ax_left = plt.subplots(figsize=figsize, dpi=dpi)
        ax_right = None

    # Gather all y-values to determine y-axis limits and ticks
    if dual_axis:
        left_ys = np.concatenate([d['y'] for d in data_list if d.get("axis", "left") == "left"])
        right_ys = np.concatenate([d['y'] for d in data_list if d.get("axis", "left") == "right"])
        combined_ys = np.concatenate([left_ys, right_ys])
    else:
        combined_ys = np.concatenate([d['y'] for d in data_list])

    (ymin, ymax), yticks = _auto_limits_and_ticks(combined_ys, max_ticks=n_ticks)

    if dual_axis:
        # Set limits and ticks for left axis
        ax_left.set_ylim([ymin, ymax])
        ax_left.set_yticks(yticks)

        # Set limits and ticks for right axis
        ax_right.set_ylim([ymin, ymax])
        ax_right.set_yticks(yticks)

        # Draw horizontal grid lines on the left axis
        for tick in yticks:
            ax_left.axhline(y=tick, color="gray", linestyle="--", linewidth=1, zorder=0)
    else:
        ax_left.set_ylim([ymin, ymax])
        ax_left.set_yticks(yticks)
        for tick in yticks:
            ax_left.axhline(y=tick, color="gray", linestyle="--", linewidth=1, zorder=0)

    # Plot each dataset
    legend_handles = []
    for i, d in enumerate(data_list):
        plot_color  = d.get("color", color_cycle[i % len(color_cycle)])
        plot_alpha  = d.get("alpha", 1)
        plot_marker = d.get("marker", ".")
        plot_axis   = d.get("axis", "left")

        if plot_axis == "left":
            ax = ax_left
        else:
            ax = ax_right

        scatter = ax.scatter(
            d['x'],
            d['y'],
            label=d.get("name", f"Series {i+1}"),
            alpha=plot_alpha,
            marker=plot_marker,
            color=plot_color
        )
        legend_handles.append(scatter)

    # Set labels and title
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(ylabel)
    if dual_axis:
        ax_right.set_ylabel(d.get("right_ylabel", "y"))  # Optional: allow different ylabel for right axis
    ax_left.set_title(title)

    # Combine legends
    if dual_axis:
        ax_left.legend(handles=legend_handles, loc="best")
        return fig, ax_left, ax_right
    else:
        ax_left.legend(handles=legend_handles, loc="best")
        return fig, ax_left
