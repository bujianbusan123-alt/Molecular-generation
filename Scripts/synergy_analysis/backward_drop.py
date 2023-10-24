# I-score
# iterative compute max I-score
import pandas as pd
import numpy as np
import copy

df = pd.read_csv('1st_2nd_onehot_prop_diff_dataset.csv')
df_pos = pd.read_csv('leaf_node_pos.csv')
cols_name_features = []
for i in list(df.columns[1:22]):
#    #if (\n not in i):
        cols_name_features.append(i)
cols_name_label = [df.columns[-2]]
df_positive = df.loc[df['D[cat] m2/s'] > 1.72e-11]
df_negative =  df.loc[df['D[cat] m2/s'] <= 1.72e-11]
df_positive['D[cat] m2/s'] = 1
df_negative['D[cat] m2/s'] = 0
df_positive = df_positive[cols_name_features[:]+cols_name_label]
df_negative = df_negative[cols_name_features[:]+cols_name_label]

#print(cols_name_features)
#print(cols_name_label)
data_positive = np.array(df_positive[cols_name_features[:]+cols_name_label])

df_all = pd.concat([df_positive, df_negative], axis=0)
df_features = df_all[cols_name_features[:]]
df_pos_features = df_pos[cols_name_features[:]]
#df_positive[cols_name_features[:]+cols_name_label]

def rank_20(df_pos, df_all, data_positive, cols_name_features, cols_name_label):
    I_score = []
    n_x_variance_2 = len(df_all)*float(np.var(np.array(df_all[cols_name_label[0]])))
    for j in np.array(df_pos_features):
        score = 0
        sample = j
        for ind, k in enumerate(list(sample)):
            if (int(k) == 1):
                #ind = list(sample).index(k)
                feature = cols_name_features[ind]
                #for i in cols_name_features:?
                x1 = len(np.array(df_all.loc[df_all['D[cat] m2/s'] >0].loc[df_all[feature] >0]))
                x2 = len(np.array(df_all.loc[df_all[feature] > 0])) * len(data_positive)/len(np.array(df_all))
                score += (x1 - x2)**2
        score = score/n_x_variance_2
        I_score.append(score)
    add_array = np.concatenate((np.array(df_pos_features), np.array(I_score).reshape(len(df_pos_features), 1)), axis=1)
    df = pd.DataFrame(add_array)
    #print(len(df_all.columns))
    df.columns = cols_name_features[:]+ ['I_score']
    # 保存all
    df_all_score = df.sort_values(by='I_score', ascending=False)
    df_all_score.to_csv('add_I_score.csv')
    #保存 20 samples
    df_20 = df.sort_values(by='I_score', ascending=False)[:20]
    df_20.to_csv('add_I_score_20.csv')
    return df_20
    #print(I_score)

# drop features for max I-score
def compute_I_score(new_array, df_all, data_positive, cols_name_features, cols_name_label):
    score = 0
    n_x_variance_2 = len(df_all)*float(np.var(np.array(df_all[cols_name_label[0]])))
    sample = new_array
    for ind, k in enumerate(list(sample)):
        if (int(k) == 1):
            #ind = list(sample).index(k)
            feature = cols_name_features[ind]
            #for i in cols_name_features:?
            x1 = len(np.array(df_all.loc[df_all['D[cat] m2/s'] >0].loc[df_all[feature] >0]))
            x2 = len(np.array(df_all.loc[df_all[feature] > 0])) * len(data_positive)/len(np.array(df_all))
            score += (x1 - x2)**2
    score = score/n_x_variance_2
    return score

def rank_max():
    # abandon features in ture for better I_score
    # score_max = 0
    dataframe = pd.read_csv('add_I_score_20.csv')
    #print(dataframe['I_score'])
    dataf = copy.deepcopy(dataframe)
    #print(len(cols_name_features))
    dataf_features = dataf[cols_name_features[:]]
    dataf = dataf[cols_name_features[:]+['I_score']]
    print(dataf.columns)
    dataf = np.array(dataf[cols_name_features[:]+['I_score']])
    #dataf = dataf[:, 1:]
    print(dataf.shape)
    for indx, new_array_1 in enumerate(np.array(dataf_features)):
        #score_max = compute_I_score(new_array_1, df_all, data_positive, cols_name_features, cols_name_label)
        score_max= dataf[indx, -1]
        bakup_array = copy.deepcopy(new_array_1)
        for ind, num in enumerate(list(new_array_1)):
        #while (compute_value >= score_max):
            if (int(num) == 1):
                bakup_array[ind] = 0
                value = compute_I_score(bakup_array, df_all, data_positive, cols_name_features, cols_name_label)
                if (value > score_max):
                    score_max = value
                else:
                    bakup_array[ind] = 1
        dataf[indx] = np.append(bakup_array, score_max)
    dataf_filter = pd.DataFrame(dataf)
    dataf_filter.columns =  cols_name_features[:] + ['I_score']
    dataf_filter.to_csv('drop_features.csv')

rank_20(df_features, df_all, data_positive, cols_name_features, cols_name_label)
#rank_max()
