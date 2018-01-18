import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lib.config.config as cfg

plt.style.available
matplotlib.style.use('seaborn-darkgrid')


def plot_learning_curves(fig_path, n_epochs, flist, name, title, style=''):
    measure = 'Acc'
    steps_measure = 'iter. (1e4) '

    plt.figure(dpi=400)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    steps = range(1, n_epochs + 1)
    plt.title(title + style)
    for ii in range(len(flist)):
        assert ii < len(cfg.FLAGS.color)
        plt.plot(steps, flist[ii], linewidth=1, color=cfg.FLAGS.color[ii], linestyle='-', marker='o',
                 markeredgecolor='black',
                 markeredgewidth=0.5, label=name[ii])

    eps = int((steps[-1]-1)/5)
    plt.xlabel(steps_measure)
    plt.xticks([0, eps, eps*2, eps*3, eps*4, eps*5], [0, 2, 4, 6, 8, 10])  # 前面一个数组表示真真实的值，后面一个表示在真实值处显示的值
    plt.ylabel(measure)
    plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.savefig(fig_path)  # 这一句要在plt.show()之前
    plt.show()
    print(fig_path)


