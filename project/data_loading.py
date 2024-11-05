import pandas as pd
import numpy as np
import chardet
from io import BytesIO
from zipfile import ZipFile
import urllib.request
from utils import val_test_split


def count_median(datadict):
    count_median = []
    for i in datadict:
        count_median.append(len(datadict[i]))
    median_size = np.median(np.array(count_median))
    return int(median_size)


def nepali_dataset_eng():
    df1 = pd.read_csv(r'../time series qua/Nepali_dataset_Eng.csv')
    df1 = df1.drop(labels=['Unnamed: 0', 'Tweet', 'Tokanize_tweet'], axis=1)
    neworder = ['Label', 'Tweet_en', 'Datetime']
    df1 = df1.reindex(columns=neworder)
    df1 = df1.rename(columns={'Label': 'label', 'Tweet_en': 'text'})
    df1 = df1[df1['label'].isin([-1, 0, 1])]
    num = df1['Datetime'].value_counts()
    sort_Num = num.sort_index()
    dates = sort_Num.index.values.tolist()
    nums = np.array(sort_Num.values.tolist())

    data_dict = {}
    pos_nums = []
    for i in range(len(dates)):
        data_dict[i] = df1[df1['Datetime'] == dates[i]].copy()  # split datasets by time
        pos_num = df1[(df1['Datetime'] == dates[i]) & (df1['label'] == 1)]['Datetime'].count()
        pos_nums.append(pos_num)
    pos_prevs = pos_nums / nums

    neg_nums = []
    for i in range(len(dates)):
        neg_num = df1[(df1['Datetime'] == dates[i]) & (df1['label'] == -1)]['Datetime'].count()
        neg_nums.append(neg_num)
    neg_prevs = neg_nums / nums

    neu_prevs = 1 - (pos_prevs + neg_prevs)

    prevalence_df = pd.DataFrame({-1: neg_prevs, 0: neu_prevs, 1: pos_prevs})

    return data_dict, prevalence_df, [-1, 0, 1], count_median(data_dict)


def global_covid19_tweets():
    training_set = r'../time series qua/global_covid19_tweet/global_covid19_tweets/Corona_NLP_train.csv'
    test_set = r'../time series qua/global_covid19_tweets/global_covid19_tweets/Corona_NLP_test.csv'
    df = pd.read_csv(training_set)
    df_test = pd.read_csv(test_set)
    df1 = pd.concat([df_test, df])
    df1 = df1.drop(labels=['UserName', 'ScreenName', 'Location'], axis=1)
    neworder = ['Sentiment', 'OriginalTweet', 'TweetAt']
    df1 = df1.reindex(columns=neworder)
    df1.loc[df1['Sentiment'] == 'Extremely Positive', 'Sentiment'] = 1
    df1.loc[df1['Sentiment'] == 'Extremely Negative', 'Sentiment'] = -1
    df1.loc[df1['Sentiment'] == 'Positive', 'Sentiment'] = 1
    df1.loc[df1['Sentiment'] == 'Negative', 'Sentiment'] = -1
    df1.loc[df1['Sentiment'] == 'Neutral', 'Sentiment'] = 0
    df1 = df1[df1['Sentiment'].isin([0, 1, -1])]
    df1 = df1.rename(columns={'Sentiment': 'label', 'OriginalTweet': 'text'})
    df1['TweetAt'] = df1['TweetAt'].str.split('-').str[1] + '-' + df1['TweetAt'].str.split('-').str[0]

    num = df1['TweetAt'].value_counts()
    sort_Num = num.sort_index()
    dates = sort_Num.index.values.tolist()
    nums = sort_Num.values.tolist()
    nums0 = np.array(nums)

    data_dict = {}
    pos_nums = []
    for i in range(len(dates)):
        data_dict[i] = df1[df1['TweetAt'] == dates[i]].copy()  # split datasets by time
        pos_num = df1[(df1['TweetAt'] == dates[i]) & (df1['label'] == 1)]['TweetAt'].count()
        pos_nums.append(pos_num)
    pos_prevs = pos_nums / nums0

    neg_nums = []
    for i in range(len(dates)):
        neg_num = df1[(df1['TweetAt'] == dates[i]) & (df1['label'] == -1)]['TweetAt'].count()
        neg_nums.append(neg_num)
    neg_prevs = neg_nums / nums0

    neu_prevs = 1 - (pos_prevs + neg_prevs)

    prevalence_df = pd.DataFrame({-1: neg_prevs, 0: neu_prevs, 1: pos_prevs})

    return data_dict, prevalence_df, [-1, 0, 1], count_median(data_dict)


def Apple_Twitter_Sentiment_DFE():
    with open(r'../time series qua/Apple-Twitter-Sentiment-DFE.csv', 'rb') as f:
        enc = chardet.detect(f.read())
    df = pd.read_csv(r'../time series qua/Apple-Twitter-Sentiment-DFE.csv', encoding=enc['encoding'])
    df1 = df.drop(labels=['_unit_id',
                          '_golden',
                          '_unit_state',
                          '_trusted_judgments',
                          '_last_judgment_at',
                          'sentiment:confidence',
                          'id',
                          'query',
                          'sentiment_gold'], axis=1)
    df1 = df1[df1['sentiment'].isin(['1', '3', '5'])]

    df1['date'] = df1['date'].str[:13]
    df1['date'] = (df1['date'].str.split(' ').str[2].astype(int) * 24 +
                   df1['date'].str.split(' ').str[3].astype(int) - 43) // 6
    neworder = ['sentiment', 'text', 'date']
    df1 = df1.reindex(columns=neworder)
    df1.loc[df1['sentiment'] == '5', 'sentiment'] = 1
    df1.loc[df1['sentiment'] == '3', 'sentiment'] = -1
    df1.loc[df1['sentiment'] == '1', 'sentiment'] = 0
    df1 = df1.rename(columns={'sentiment': 'label'})

    num = df1['date'].value_counts()
    sort_Num = num.sort_index()

    dates = sort_Num.index.tolist()

    nums = sort_Num.values.tolist()
    nums0 = np.array(nums)

    '''
    Prevalence of positives
    '''
    data_dict = {}
    pos_nums = []
    for i in range(len(dates)):
        data_dict[i] = df1[df1['date'] == dates[i]].copy()  # split datasets by time
        pos_num = df1[(df1['date'] == dates[i]) & (df1['label'] == 1)]['date'].count()
        pos_nums.append(pos_num)
    pos_prevs = pos_nums / nums0

    '''
    Prevalence of negatives
    '''
    neg_nums = []
    for i in range(len(dates)):
        neg_num = df1[(df1['date'] == dates[i]) & (df1['label'] == -1)]['date'].count()
        neg_nums.append(neg_num)
    neg_prevs = neg_nums / nums0

    '''
    prevalence of neutrals
    '''
    neu_prevs = 1 - (pos_prevs + neg_prevs)

    '''
    collect prevalence of all labels
    '''
    prevalence_df = pd.DataFrame({-1: neg_prevs, 0: neu_prevs, 1: pos_prevs})

    return data_dict, prevalence_df, [-1, 0, 1], count_median(data_dict)


def bike():
    training_size = 38
    # url = urllib.request.urlopen(
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip")
    #
    # my_zip_file = ZipFile(BytesIO(url.read()))
    # print(my_zip_file.namelist())
    # f = my_zip_file.namelist()[2]
    f = r'../time series qua/bike_sharing_dataset/hour.csv'
    dta = pd.read_csv(f, header=0, skipinitialspace=True)
    dta = dta.drop(["instant", "casual", "registered"], axis=1)
    dta = pd.get_dummies(dta, columns=["season", "yr", "mnth", "hr", "weekday", "weathersit"])

    bins = [0, 100, 1000]
    labels = [0, 1]
    dta['cnt'] = pd.cut(dta['cnt'], bins=bins, labels=labels)
    dta['cnt'] = dta['cnt'].astype("int64")
    dta = dta.rename(columns={'cnt': 'label'})

    binned = False
    if binned:
        for col in list(dta)[2:6]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

    num = dta['dteday'].value_counts()
    dates = num.sort_index().index.values.tolist()

    x = 2
    df_date_index = dta.copy()
    for i, date in enumerate(dates):
        df_date_index.loc[df_date_index['dteday'] == date, 'dteday'] = i // x

    _ = df_date_index['dteday'].value_counts().sort_index()
    timestamps = _.index.values.tolist()
    amounts = np.array(_.values.tolist())

    data_dict = {}
    pos_nums = []
    for i, t in enumerate(timestamps):
        data_dict[i] = df_date_index[df_date_index['dteday'] == t].copy()
        del data_dict[i]['dteday']

        pos_num = df_date_index[(df_date_index['dteday'] == t) & (df_date_index['label'] == 1)]['dteday'].count()
        pos_nums.append(pos_num)
    pos_nums = np.array(pos_nums)
    pos_prevs = pos_nums / amounts

    prevalence_df = pd.DataFrame({0: 1 - pos_prevs, 1: pos_prevs})
    training_data, ts_data_dict, ts_prevalence = val_test_split(data_dict.copy(), prevalence_df, training_size)
    training_data = training_data.sample(frac=1.0, replace=False, random_state=42).reset_index(drop=True)

    return training_data, ts_data_dict, ts_prevalence, [0, 1], None


def energy():
    training_size = 15
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"

    dta = pd.read_csv(url, header=0, skipinitialspace=True)
    dta = dta.drop(["rv1", "rv2"], axis=1)
    dta.Appliances.describe()

    bins = [0, 50, 100, 2000]
    labels = [-1, 0, 1]
    dta['Appliances'] = pd.cut(dta['Appliances'], bins=bins, labels=labels)
    dta['Appliances'] = dta['Appliances'].astype("int64")
    dta = dta.rename(columns={'Appliances': 'label'})

    binned = False
    if binned:
        for col in list(dta)[1:]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

    dta['date'] = dta['date'].str[:10]
    _ = dta['date'].value_counts().sort_index()

    timestamps = _.index.values.tolist()
    amounts = np.array(_.values.tolist())

    data_dict = {}
    pos_nums = []
    for i, t in enumerate(timestamps):
        data_dict[i] = dta[dta['date'] == t].copy()
        del data_dict[i]['date']

        pos_num = dta[(dta['date'] == t) & (dta['label'] == 1)]['date'].count()
        pos_nums.append(pos_num)
    pos_nums = np.array(pos_nums)
    pos_prevs = pos_nums / amounts

    neg_nums = []
    for i, t in enumerate(timestamps):
        neg_num = dta[(dta['date'] == t) & (dta['label'] == -1)]['date'].count()
        neg_nums.append(neg_num)
    neg_prevs = neg_nums / amounts

    neu_prevs = 1 - (pos_prevs + neg_prevs)

    prevalence_df = pd.DataFrame({-1: neg_prevs, 0: neu_prevs, 1: pos_prevs})
    training_data, ts_data_dict, ts_prevalence = val_test_split(data_dict.copy(), prevalence_df, training_size)
    training_data = training_data.sample(frac=1.0, replace=False, random_state=42).reset_index(drop=True)

    return training_data, ts_data_dict, ts_prevalence, [-1, 0, 1], None


def news():
    training_size = 21
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    dta_file = my_zip_file.namelist()[2]

    dta = pd.read_csv(my_zip_file.open(dta_file), skipinitialspace=True)
    dta = dta.drop(columns="url")

    bins = [0, 1000, 1000000]
    labels = [0, 1]
    dta['shares'] = pd.cut(dta['shares'], bins=bins, labels=labels)
    dta['shares'] = dta['shares'].astype("int64")
    dta = dta.rename(columns={'shares': 'label'})

    binned = False
    if binned:
        qcols = ['timedelta',
                 'n_tokens_title',
                 'n_tokens_content',
                 'n_unique_tokens',
                 'n_non_stop_words',
                 'n_non_stop_unique_tokens',
                 'num_hrefs',
                 'num_self_hrefs',
                 'num_imgs',
                 'num_videos',
                 'average_token_length',
                 'num_keywords',
                 'kw_min_min',
                 'kw_max_min',
                 'kw_avg_min',
                 'kw_min_max',
                 'kw_max_max',
                 'kw_avg_max',
                 'kw_min_avg',
                 'kw_max_avg',
                 'kw_avg_avg',
                 'self_reference_min_shares',
                 'self_reference_max_shares',
                 'self_reference_avg_sharess',
                 'LDA_00',
                 'LDA_01',
                 'LDA_02',
                 'LDA_03',
                 'LDA_04',
                 'global_subjectivity',
                 'global_sentiment_polarity',
                 'global_rate_positive_words',
                 'global_rate_negative_words',
                 'rate_positive_words',
                 'rate_negative_words',
                 'avg_positive_polarity',
                 'min_positive_polarity',
                 'max_positive_polarity',
                 'avg_negative_polarity',
                 'min_negative_polarity',
                 'max_negative_polarity',
                 'title_subjectivity',
                 'title_sentiment_polarity',
                 'abs_title_subjectivity',
                 'abs_title_sentiment_polarity']
        for col in qcols:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

    num = dta['timedelta'].value_counts()
    dates = num.sort_index().index.values.tolist()

    x = 3
    df_date_index = dta.copy()
    for i, date in enumerate(dates):
        df_date_index.loc[df_date_index['timedelta'] == date, 'timedelta'] = i // x

    _ = df_date_index['timedelta'].value_counts().sort_index()
    timestamps = _.index.values.tolist()
    amounts = np.array(_.values.tolist())

    data_dict = {}
    pos_nums = []
    for i, t in enumerate(timestamps):
        data_dict[i] = df_date_index[df_date_index['timedelta'] == t].copy()
        del data_dict[i]['timedelta']

        pos_num = df_date_index[(df_date_index['timedelta'] == t) & (df_date_index['label'] == 1)]['timedelta'].count()
        pos_nums.append(pos_num)
    pos_nums = np.array(pos_nums)
    pos_prevs = pos_nums / amounts

    prevalence_df = pd.DataFrame({0: 1 - pos_prevs, 1: pos_prevs})
    training_data, ts_data_dict, ts_prevalence = val_test_split(data_dict.copy(), prevalence_df, training_size)
    training_data = training_data.sample(frac=1.0, replace=False, random_state=42).reset_index(drop=True)

    return training_data, ts_data_dict, ts_prevalence, [0, 1], None


def loading(dataname):

    if dataname == 'nepali_dataset_eng':
        return nepali_dataset_eng()

    elif dataname == 'global_covid19_tweets':
        return global_covid19_tweets()

    elif dataname == 'Apple-Twitter-Sentiment-DFE':
        return Apple_Twitter_Sentiment_DFE()

    elif dataname == 'bike':
        return bike()

    elif dataname == 'energy':
        return energy()

    elif dataname == 'news':
        return news()
