import matplotlib.pyplot as plot

from config import PLOT_PATH


def get_plot_file_path(name, plot_path):
    return f"{plot_path}/{name}"


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
    plot.rcParams["figure.figsize"] = [12, 6]
    plot.scatter(x_data, y_data)
    plot.xticks(x_data, [str(i) for i in x_data], rotation=90)
    create_plot(title, x_label, y_label, plot_name)


def save_worker_data(y_data, name):
    with open(f"{PLOT_PATH}/data_{name}", "w") as f:
        f.write("\n".join([str(num) for num in y_data]))


def save_test_data(x_data, returns, updates=0):
    with open(
        f"{PLOT_PATH}/plot_test_data_{updates if updates else 'final'}.txt", "w"
    ) as f:
        for name, data in zip(x_data, returns):
            f.write(f"{name}:\n")
            f.write("\n".join([str(num) for num in data]))
            f.write("\n")
