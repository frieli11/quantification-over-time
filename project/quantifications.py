import numpy as np
from quantifiers import DyS, ACC, GPAC, EDy
from sklearn.metrics import confusion_matrix
from classification import Classifying
import pandas as pd
from utils import mse, mae
import os


def ACC_on_TSsets(val_set, test_set_dict, senti_model, classes):
    val_y_res = Classifying.analyzer(val_set, senti_model, classes)
    val_y_ = val_y_res[0]

    tests_y_ = {}
    for i in range(len(test_set_dict)):
        test_y_ = Classifying.analyzer(test_set_dict[i], senti_model, classes)[0]
        tests_y_[i] = test_y_

    qtfied_distribution = []
    for i, cla in enumerate(classes):
        val_y_one = val_y_.copy()

        val_y_one.loc[val_y_one['true_y'] == cla, 'true_y'] = 'True'
        val_y_one.loc[val_y_one['true_y'].isin(classes), 'true_y'] = 'False'
        val_y_one.loc[val_y_one['pred_y'] == cla, 'pred_y'] = 'True'
        val_y_one.loc[val_y_one['pred_y'].isin(classes), 'pred_y'] = 'False'
        tn, fp, fn, tp = confusion_matrix(val_y_one['true_y'], val_y_one['pred_y']).ravel()
        tpr = tp / (tp+fn)
        fpr = fp / (fp+tn)

        qtfied_prevs_one = []
        for j in range(len(test_set_dict)):
            test_y_one = tests_y_[j]['pred_y'].copy()
            test_y_one[test_y_one == cla] = 'True'
            test_y_one[test_y_one.isin(classes)] = 'False'
            qua_prev = ACC(test_y_one, tpr, fpr)
            qtfied_prevs_one.append(qua_prev)
        qtfied_distribution.append(qtfied_prevs_one)

    qtfied_distribution = np.array(qtfied_distribution).T
    qtfied_distribution[np.all(qtfied_distribution == 0, axis=1)] = 1
    qtfied_distribution = qtfied_distribution / (np.sum(qtfied_distribution, axis=1).reshape(-1, 1))
    return qtfied_distribution


def DyS_on_TSsets(val_set, test_set_dict, senti_model, classes):
    val_y_res = Classifying.analyzer(val_set, senti_model, classes)
    val_score = val_y_res[1]

    tests_scores = {}
    for i in range(len(test_set_dict)):
        test_score = Classifying.analyzer(test_set_dict[i], senti_model, classes)[1]
        tests_scores[i] = test_score

    qtfied_distribution = []
    for i, cla in enumerate(classes):
        val_score_one = val_score[val_score['true_y'] == cla][cla]
        val_score_rest = val_score[val_score['true_y'] != cla][cla]

        qtfied_prevs_one = []
        for j in range(len(test_set_dict)):
            test_score_one = tests_scores[j][cla]
            qua_prev = DyS(val_score_one, val_score_rest, test_score_one, measure='topsoe')
            qtfied_prevs_one.append(qua_prev)
        qtfied_distribution.append(qtfied_prevs_one)

    qtfied_distribution = np.array(qtfied_distribution).T
    qtfied_distribution[np.all(qtfied_distribution == 0, axis=1)] = 1
    qtfied_distribution = qtfied_distribution / (np.sum(qtfied_distribution, axis=1).reshape(-1, 1))

    return qtfied_distribution


def GPAC_on_TSsets(val_set, test_set_dict, senti_model, classes):
    val_res = Classifying.analyzer(val_set, senti_model, classes)
    val_scores = val_res[1].iloc[:, :-1].to_numpy()
    val_labels = val_res[0]

    tests_scores = {}
    for i in range(len(test_set_dict)):
        test_score = Classifying.analyzer(test_set_dict[i], senti_model, classes)[1]
        tests_scores[i] = test_score.iloc[:, :-1].to_numpy()

    qtfied_distribution = []
    for j in range(len(test_set_dict)):
        qua_prev = GPAC(val_scores,
                        tests_scores[j],
                        val_labels['true_y'].to_numpy(),
                        classes)
        qtfied_distribution.append(qua_prev)

    qtfied_distribution = np.array(qtfied_distribution)

    return qtfied_distribution


def EDy_on_TSsets(val_set, test_set_dict, senti_model, classes):
    val_res = Classifying.analyzer(val_set, senti_model, classes)
    val_scores = val_res[1].iloc[:, :-1].to_numpy()
    val_labels = val_res[0]

    tests_scores = {}
    for i in range(len(test_set_dict)):
        test_score = Classifying.analyzer(test_set_dict[i], senti_model, classes)[1]
        tests_scores[i] = test_score.iloc[:, :-1].to_numpy()

    qtfied_distribution = []
    for j in range(len(test_set_dict)):
        qua_prev = EDy(val_scores,
                       val_labels['true_y'].to_numpy(),
                       tests_scores[j],
                       classes)
        qtfied_distribution.append(qua_prev)

    qtfied_distribution = np.array(qtfied_distribution)

    return qtfied_distribution


def CC_on_TSsets(test_set_dict, senti_model, classes):
    tests_y_ = {}
    for i in range(len(test_set_dict)):
        test_y_ = Classifying.analyzer(test_set_dict[i], senti_model, classes)[0]
        tests_y_[i] = test_y_

    qtfied_distribution = []
    for j in range(len(test_set_dict)):
        qua_prev = []
        s = len(tests_y_[j])
        for c in classes:
            n = tests_y_[j][tests_y_[j]['pred_y'] == c]['pred_y'].count()
            qua_prev.append(n/s)

        qtfied_distribution.append(qua_prev)

    qtfied_distribution = np.array(qtfied_distribution)

    return qtfied_distribution


def getMAE_val_set(val_set, qua, mod, c, data, name, random_seed):
    subsamples_dict = {}
    subsamples_dsts = []
    for i in range(name[1]):
        subsamples_dict[i] = data[i]
        s = subsamples_dict[i].value_counts('label').sum()
        p = []
        for label in c:
            p.append(subsamples_dict[i][subsamples_dict[i]['label'] == label]['label'].count() / s)
        subsamples_dsts.append(p)

    subsamples_prevs = np.array(subsamples_dsts)

    val_MAE, val_MSE, sep_MAE, qtfd_dsts = None, None, None, None

    if qua == 'DyS':
        qtfd_dsts = DyS_on_TSsets(val_set, subsamples_dict, mod, c)
    elif qua == 'ACC':
        qtfd_dsts = ACC_on_TSsets(val_set, subsamples_dict, mod, c)
    elif qua == 'GPAC':
        qtfd_dsts = GPAC_on_TSsets(val_set, subsamples_dict, mod, c)
    elif qua == 'EDy':
        qtfd_dsts = EDy_on_TSsets(val_set, subsamples_dict, mod, c)
    elif qua == 'CC':
        qtfd_dsts = CC_on_TSsets(subsamples_dict, mod, c)
    elif qua == 'ReadMe2':
        qtfd_dsts = np.array(pd.read_csv(f'./ReadMe_Implement/data/{name[0]}/seed{random_seed}/val_preds.csv'))

    sep_MAE = abs(subsamples_prevs - qtfd_dsts).sum(axis=0) / len(subsamples_dsts)
    val_MAE = mae(subsamples_prevs, qtfd_dsts)
    val_MSE = mse(subsamples_prevs, qtfd_dsts)
    return val_MAE, val_MSE, sep_MAE, qtfd_dsts


def qtfied_dists(valset, data_dict, dataname, qua, mod, c, random_seed):

    try:
        quantified_dsts = pd.read_csv(
            rf'.\quant_results\_{dataname[0]}\{qua}-{str(mod)[:6]}-{dataname[0]}-{dataname[1]}.csv').drop(
            labels=['Unnamed: 0'], axis=1).to_numpy()
        quantified_dsts = np.nan_to_num(quantified_dsts, nan=1 / len(c))

    except IOError:
        quantified_dsts = 0
        if qua == 'DyS':
            quantified_dsts = DyS_on_TSsets(valset, data_dict, mod, c)
        elif qua == 'ACC':
            quantified_dsts = ACC_on_TSsets(valset, data_dict, mod, c)
        elif qua == 'GPAC':
            quantified_dsts = GPAC_on_TSsets(valset, data_dict, mod, c)
        elif qua == 'EDy':
            quantified_dsts = EDy_on_TSsets(valset, data_dict, mod, c)
        elif qua == 'CC':
            quantified_dsts = CC_on_TSsets(data_dict, mod, c)
        elif qua == 'ReadMe2':
            df_quantified_dsts = pd.read_csv(f'./ReadMe_Implement/data/{dataname[0]}/seed{random_seed}/test_preds.csv')
            quantified_dsts = np.array(df_quantified_dsts)

        quantified_dsts = np.nan_to_num(quantified_dsts, nan=1 / len(c))

        pd_quantified_dsts = {}
        for i, cls in enumerate(c):
            pd_quantified_dsts[cls] = quantified_dsts[:, i]
        pd_quantified_dsts = pd.DataFrame(quantified_dsts)
        dataset_folder = rf'.\quant_results\_{dataname[0]}'
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        pd_quantified_dsts.to_csv(dataset_folder+rf'\{qua}-{str(mod)[:6]}-{dataname[0]}.csv')

    return quantified_dsts
