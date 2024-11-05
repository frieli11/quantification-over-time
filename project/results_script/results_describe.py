import pandas as pd
import numpy as np


seeds = [1,
         # 2,
         # 3,
         # 4,
         # 5,
         # 6,
         # 7,
         # 8,
         # 9,
         # 10
         ]

# TSF = ['QFY', 'MA', 'WAUA', 'KFUA', 'GPR', 'KFGPR']
TSF = ['QFY', 'MA', 'WAUA', 'KFUA', 'KFMA', 'GPR']

# tot = np.zeros(72*6).reshape(72, 6)
tot = np.zeros(3*4*5*6).reshape(3*4*5, 6)
for seed in seeds:
    res = pd.read_csv(r'../Output_Files/MAE results seed '+str(seed)+'.csv').drop(labels=['Unnamed: 0'], axis=1)
    res = res.to_numpy()[:, 3:9]
    tot = tot + res
tot = tot / len(seeds)

tot_res = pd.read_csv(r'../Output_Files/MAE results seed '+str(1)+'.csv').iloc[:, :4]
for i, m in enumerate(TSF):
    tot_res[m] = tot[:, i]

# Save quanti_results of one random case in to a table
TSF_results = tot_res.iloc[:, 4:10]
best_m = []
for i in range(len(TSF_results)):
    mini = 1
    m_num = -1
    for col_num, cell in enumerate(TSF_results.iloc[i, :]):
        if cell < mini:
            mini = cell
            m_num = col_num
    best_m.append(TSF[m_num])

# # win_counts = {'QFY': [0], 'MA': [0], 'WAUA': [0], 'KFUA': [0], 'GPR': [0], 'KFGPR': [0]}
# win_counts = {'QFY': [0], 'MA': [0], 'WAUA': [0], 'KFUA': [0], 'KFMA': [0], 'GPR': [0]}
# for m in best_m:
#     win_counts[m][0] = win_counts[m][0] + 1
# df_win_counts = pd.DataFrame(win_counts)
# df_win_counts.to_csv('../Output_Files/win_counts.csv')
#
# pairs = []
# for m_i in range(len(TSF)-1, -1, -1):
#     if m_i == 0:
#         break
#     for m_j in range(m_i-1, -1, -1):
#         pair_win_counts = {TSF[m_i]: 0, TSF[m_j]: 0}
#         for row in range(len(TSF_results)):
#             if TSF_results.iloc[row, m_i] > TSF_results.iloc[row, m_j]:
#                 pair_win_counts[TSF[m_j]] = pair_win_counts[TSF[m_j]] + 1
#             elif TSF_results.iloc[row, m_i] < TSF_results.iloc[row, m_j]:
#                 pair_win_counts[TSF[m_i]] = pair_win_counts[TSF[m_i]] + 1
#             else:
#                 pair_win_counts[TSF[m_i]] = pair_win_counts[TSF[m_i]] + 1
#                 pair_win_counts[TSF[m_j]] = pair_win_counts[TSF[m_j]] + 1
#         pairs.append([TSF[m_i], TSF[m_j], pair_win_counts[TSF[m_i]], pair_win_counts[TSF[m_j]]])
# pairs = np.array(pairs)
# pair_wins = pd.DataFrame({'method 1': pairs[:, 0],
#                           'method 2': pairs[:, 1],
#                           'method 1 wins': pairs[:, 2],
#                           'method 2 wins': pairs[:, 3]})
# pair_wins.to_csv('../Output_Files/pair_wins.csv')

tot_res['best_method'] = np.array(best_m)
tot_res = tot_res.drop(labels=['Unnamed: 0'], axis=1)
tot_res.to_csv('../Output_Files/MAE quanti_results mean.csv')
