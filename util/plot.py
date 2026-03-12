import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_inference_result(sensor_sample, prompt_text, answer_text, title=None, channel_names=None):
    """
    Return a matplotlib figure visualizing time series (left) and prompt + answer (right).

    Args:
        sensor_sample (torch.Tensor): shape [1, C, L] or [C, L]
        prompt_text (str): prompt used during inference
        answer_text (str): model-generated answer
        title (str): optional title
        channel_names (list[str]): optional list of channel names (len = C)

    Returns:
        fig (matplotlib.figure.Figure): figure object (not shown)
    """
    if sensor_sample.ndim == 3:
        sensor_sample = sensor_sample.squeeze(0)  # [C, L]
    elif sensor_sample.ndim != 2:
        raise ValueError("Expected sensor_sample shape [1, C, L] or [C, L]")

    C, L = sensor_sample.shape
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(C)]

    # Dynamic figure height
    fig_height = max(6, 1.5 * C)
    fig = plt.figure(figsize=(14, fig_height), constrained_layout=True)
    gs = gridspec.GridSpec(nrows=C, ncols=2, width_ratios=[3.5, 2], figure=fig)

    # Time series subplot
    for i in range(C):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(sensor_sample[i].cpu().numpy(), linewidth=1.0)
        ax.set_xlim(0, L)
        ax.set_ylabel(channel_names[i], fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        if i < C - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (minutes)", fontsize=10)

    # Text content subplot
    text_ax = fig.add_subplot(gs[:, 1])
    text_ax.axis("off")
    full_text = f"Prompt:\n{prompt_text}\n\nGenerated Answer:\n{answer_text}"
    text_ax.text(
        0.01,
        0.99,
        full_text,
        va="top",
        ha="left",
        wrap=True,
        fontsize=10,
        fontfamily="monospace",
        transform=text_ax.transAxes,
    )

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    return fig  # Do NOT call plt.show()