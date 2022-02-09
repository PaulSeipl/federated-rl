import matplotlib.pyplot as plot


def create_plot(title, x_label, y_label, plot_name):
    file_types = ["svg", "pdf", "png"]
    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    for file_type in file_types:
        plot.savefig(
            f"{plot_name}.{file_type}", format=file_type, dpi=300, bbox_inches="tight"
        )
    plot.close()


def create_worker_plot(
    title, x_data, x_label, y_data, y_label, interval, training_episodes, plot_name
):
    plot.rcParams["figure.figsize"] = [12, 6]
    plot.plot(x_data, y_data, linewidth=0.5)
    vertical_lines = [lap * training_episodes - 1 for lap in range(1, interval)]
    for vertical_line in vertical_lines:
        plot.axvline(
            x=vertical_line, alpha=0.5, color="r", linestyle="-", linewidth=0.5
        )
    create_plot(title, x_label, y_label, plot_name)


def create_main_plot(title, x_data, x_label, y_data, y_label, plot_name):
    plot.rcParams["figure.figsize"] = [6, 10]
    plot.plot(x_data, y_data)
    create_plot(title, x_label, y_label, plot_name)
