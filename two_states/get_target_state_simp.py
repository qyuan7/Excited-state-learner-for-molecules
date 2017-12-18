#!/usr/bin/env python
"""
Get the first two excited states with oscillator strength larger than 0.01
"""
import pandas as pd
import numpy as np


def read_exp_result(csv_file):
    exp_df = pd.read_csv(csv_file)
    #exp_df.set_index(['Compound'], inplace=True)
    exp_df['Stability'] = (exp_df['ISO LOGT50'] <= 3.60).astype(int)
    return exp_df 

def get_all_target_states(filename, two_states=False):
    my_cols = range(40)
    data = pd.read_csv(filename, sep=' ', header=None, names=my_cols, engine='python')
    data1 = data.iloc[:,0:5]
    data1.columns = ['Compound', 'e1','f1','e2','f2']
    if two_states:
        return data1
    data1['energy'] = np.where((data1['f2']>data1['f1']),data1['e2'],data1['e1'])
    data1['strength'] = np.where((data1['f2']>data1['f1']),data1['f2'],data1['f1'])
    data2 = data1[['Compound','energy','strength']]
    return data2

def main():     
    state_data = get_all_target_states('ex_properties', two_states=True)
    exp_df = read_exp_result('strobi_iso_exp_data.csv')
    train_test_sep = pd.read_csv('strobi_iso_train_vs_test.csv')
    total_df = pd.merge(state_data, exp_df, on=['Compound'])
    total_df = pd.merge(total_df, train_test_sep, on=['Compound'])
    train_df = total_df.loc[total_df['Group'] == 'train']
    test_df = total_df.loc[total_df['Group'] == 'test']
    train_df.to_csv('train_two_target_states', sep=',',encoding='utf-8', index=False)
    test_df.to_csv('test_two_target_states', sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    main()
