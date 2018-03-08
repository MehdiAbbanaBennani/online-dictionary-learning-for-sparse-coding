import matplotlib.pyplot as plt


def plot_many(x, y_list, x_label, y_label, title=None, filename=None, to_save=False):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)

    for i in range(len(y_list)) :
        plt.plot(x, y_list[i])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='best')

    if to_save:
        fig.savefig("logs/" + filename + ".png")
    else:
        plt.show()