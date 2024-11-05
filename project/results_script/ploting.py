import matplotlib.pyplot as plt
import seaborn as sns
import os

def plots(true_dsts, quantified_dsts, modified_dsts, dataset, head, condition, c, qua, tsa, show=True):
    fig, _ = plt.subplots(len(c), 1, figsize=(10, 6))
    fig.suptitle(head + condition)
    x = [j for j in range(true_dsts.shape[0])]

    # # slice cut
    # x = [j for j in range(true_dsts.shape[0]-255)]
    # true_dsts = true_dsts[40:-215, :]
    # quantified_dsts = quantified_dsts[40:-215, :]
    # modified_dsts = modified_dsts[40:-215, :]

    # x = [j for j in range(true_dsts.shape[0] - 285)]
    # true_dsts = true_dsts[65:-220, :]
    # quantified_dsts = quantified_dsts[65:-220, :]
    # modified_dsts = modified_dsts[65:-220, :]


    for i in range(len(c)):
        if i != len(c)-1:
            orig = _[i].plot(x, true_dsts[:, i], color='black', label='True')
            quant = _[i].plot(x,  quantified_dsts[:, i], color='blue', label=qua)
            combi = _[i].plot(x, modified_dsts[:, i], color='orange', label=f'{qua}+{tsa}')
            # combi = _[i].plot(x, modified_dsts[:, i], color='orange', label='TSA')
            _[i].set_ylabel('Prevalence of '+str(c[i]))
            _[i].tick_params('x', labelbottom=False)
            _[i].legend(loc='upper right')
            _[i].grid()

        else:
            orig = _[i].plot(x, true_dsts[:, i], color='black', label='True')
            quant = _[i].plot(x,  quantified_dsts[:, i], color='blue', label=qua)
            combi = _[i].plot(x, modified_dsts[:, i], color='orange', label=f'{qua}+{tsa}')
            # combi = _[i].plot(x, modified_dsts[:, i], color='orange', label='TSA')
            _[i].set_ylabel('Prevalence of '+str(c[i]))
            _[i].legend(loc='upper right')
            _[i].set_xlabel('Timestamp Unit')
            _[i].grid()

    # plt.subplots_adjust(left=0.07, right=0.975, top=0.96, bottom=0.1)  # adjust the blank

    if not os.path.exists(rf'.\plots\_{dataset}'):
        os.makedirs(rf'.\plots\_{dataset}')

    plt.savefig(rf'.\plots\_{dataset}\{head[16:-15]}-{condition}.png', dpi=150)
    # plt.savefig(rf'.\plots\_{dataset}\{head[16:-15]}-{condition}.eps', format='eps')

    if show:
        plt.show(block=False)

    return [x, true_dsts[:, 0], quantified_dsts[:, 0], modified_dsts[:, 0]]
