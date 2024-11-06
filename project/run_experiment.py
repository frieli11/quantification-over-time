import pandas as pd
import numpy as np
import data_loading
import utils
from classification import trainingModel
import quantifications as qfy
from time_series_adjustment import MovingAverage, KalmanMA
from utils import params_KFMA
import argparse


seeds = [1, 2,
         3, 4, 5, 6, 7, 8, 9, 10
         ]
text_senti_data = [('global_covid19_tweets', 15), ('nepali_dataset_eng', 15), ('Apple-Twitter-Sentiment-DFE', 15)]
tubular_data = [('bike', 55),
                ('energy', 20), ('news', 36)
                ]
classifiers_set1 = ['LR', 'RF']
classifiers_set2 = ['vader', 'amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061']
qua_methods = ['DyS', 'ACC', 'GPAC', 'EDy']
TSA_methods = ['QFY', 'MA', 'KFMA']
unified_window = 4


def experiment(dataset, classifier, quantifier, tsa, random_state):

    print(f'-----{dataset[0]}-{classifier}-{quantifier}-{tsa}-{random_state}-----')

    if dataset in tubular_data:
        training_set, ts_chunks, ts_prevalence, c, t_size = data_loading.loading(dataset[0])
        classifier = trainingModel.trainer(training_set.loc[:, ~training_set.columns.isin(['label'])],
                                           training_set['label'], classifier, random_state)
    else:
        ts_chunks, ts_prevalence, c, t_size = data_loading.loading(dataset[0])

    '''
    Majority of quantification methods may need datasets to be split for a validation
    set and test sets since they need the scores from labelled datasets to do
    distribution matching or TPR, FPR. The weights of Time Series Forecast and
    Quantification also need information of sentiment classifier gained by analyzing
    validation set.
    '''

    '''
    ----------------------------------initial values-----------------------------------
    '''
    if dataset[1] < unified_window:
        lf = 0
    else:
        lf = dataset[1] - unified_window
    inital_value = ts_prevalence.iloc[lf:dataset[1], :].to_numpy()

    '''
    ----------------------------------validation set----------------------------------
    '''
    val_true = ts_prevalence[:dataset[1]].to_numpy()
    val_set, test_sets, test_dsts = utils.val_test_split(ts_chunks.copy(), ts_prevalence, dataset[1])

    '''
    Whether use sets with natural distributions or synthetic distribution.
    'utils.getMAE_val_set2()' does not use the random seed. because the val 
    subsets are real datasets of different timestamps, not synthetic sampled 
    from validation set.
    '''
    val_MAE, val_MSE, sep_mae, val_pred_dists = qfy.getMAE_val_set(val_set, quantifier, classifier, c,
                                                                   ts_chunks, dataset,
                                                                   random_seed=random_state)
    print('MAE on val samples:', round(val_MAE, 4))

    '''
    -------------------------------quantifying test sets-------------------------------
    '''
    quantified_dsts = qfy.qtfied_dists(val_set, test_sets, dataset, quantifier, classifier, c, random_seed=random_state)
    Qua_MAE = utils.mae(test_dsts, quantified_dsts)

    if tsa == 'QFY':
        print(f'***{quantifier} MAE***:', round(Qua_MAE, 4))
        return Qua_MAE

    else:
        modified_dsts = []
        val_init_value = np.empty((0, len(c)))
        validation = [val_init_value, val_pred_dists, val_true, c, unified_window]
        if tsa == 'MA':
            for index in range(len(c)):
                modified_prevs = MovingAverage(initial_value=inital_value[:, index],
                                               quantified_prevs=quantified_dsts[:, index],
                                               window=unified_window)
                modified_dsts.append(modified_prevs)
                if len(c) == 2:
                    modified_dsts.append(1 - modified_prevs)
                    break

        elif tsa == 'KFMA':
            _, _q = params_KFMA(validation, val_MSE)
            q = 10 ** _q
            for index in range(len(c)):
                modified_prevs = KalmanMA(initial_value=inital_value[:, index],
                                          observations=quantified_dsts[:, index],
                                          qtfy_error=val_MSE,
                                          state_dim=unified_window,
                                          q=q)
                modified_dsts.append(modified_prevs)

        modified_dsts = np.array(modified_dsts).T
        modified_dsts = modified_dsts / (np.sum(modified_dsts, axis=1).reshape(-1, 1))

        Combi_MAE = utils.mae(test_dsts, modified_dsts)
        print(f'***{quantifier}+{tsa} MAE***:', round(Combi_MAE, 4))
        return Combi_MAE


def qot(data_format):
    if data_format == 'tabular':
        dataset_set = tubular_data
        clsfiers = classifiers_set1
    else:
        dataset_set = text_senti_data
        clsfiers = classifiers_set2

    seed_tables = []
    for seed in seeds:
        idx = 0
        outputfile = pd.DataFrame({'Dataset': [],
                                   'QuaMethod': [],
                                   'Classifier': [],
                                   'QFY': [],
                                   'MA': [],
                                   'KFMA': []})

        for data_name in dataset_set:
            for qua in qua_methods:
                for mod in clsfiers:
                    output = [data_name[0], qua, mod]
                    for tsa_method in TSA_methods:
                        res = experiment(data_name, mod, qua, tsa_method, seed)
                        output.append(res)
                    outputfile.loc[idx] = output
                    idx = idx + 1

        seed_tables.append(outputfile)

    tot = np.zeros(3 * 4 * 2 * 3).reshape(3 * 4 * 2, 3)
    for i in range(len(seeds)):
        res = seed_tables[i].to_numpy()[:, 3:6]
        tot = tot + res
    tot = tot / len(seeds)

    tot_res = seed_tables[1].iloc[:, :3]
    for i, m in enumerate(TSA_methods):
        tot_res[m] = tot[:, i]

    # Save quanti_results of one random case in to a table
    TSF_results = tot_res.iloc[:, 3:6]
    best_m = []
    for i in range(len(TSF_results)):
        mini = 1
        m_num = -1
        for col_num, cell in enumerate(TSF_results.iloc[i, :]):
            if cell < mini:
                mini = cell
                m_num = col_num
        best_m.append(TSA_methods[m_num])

    tot_res['best_method'] = np.array(best_m)
    tot_res.to_csv(f'output_files/MAE_quanti_results_mean_{data_format}.csv')


def sota_qot():
    implementation = [['amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061', 'CC', 'QFY'],
                      ['amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061', 'CC', 'MA'],
                      ['None', 'ReadMe2', 'QFY'],
                      ['None', 'ReadMe2', 'KFMA']]

    outputfile = pd.DataFrame({'QoT Method': [text_senti_data[0][0], text_senti_data[1][0], text_senti_data[2][0]]})

    tot = np.zeros((3, 4))
    for seed in seeds:
        for i, data_name in enumerate(text_senti_data):
            for j, cond in enumerate(implementation):
                res = experiment(data_name, cond[0], cond[1], cond[2], seed)
                tot[i, j] += res
    tot = tot / len(seeds)

    outputfile['CC'] = tot[:, 0]
    outputfile['CC+MA'] = tot[:, 1]
    outputfile['ReadMe2'] = tot[:, 2]
    outputfile['ReadMe2+KFMA'] = tot[:, 3]

    outputfile.to_csv(f'output_files/sota_qot_MAE_quanti_results_mean.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run',
                        choices=['tubular', 'textual', 'sota_qot'],
                        default='blue',
                        help='Choose experiment')
    args = parser.parse_args()
    if args.run == 'sota_qot':
        sota_qot()
    else:
        qot(args)

    # qot('textual')
    # qot('tabular')
    # sota_qot()

