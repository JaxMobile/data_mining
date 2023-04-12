import py_stringsimjoin as ssj
import py_stringmatching as sm
import pandas as pd

table_A_link = 'https://raw.githubusercontent.com/JaxMobile/data_mining/master/datasets/abtProfiles.csv'
table_B_link = 'https://raw.githubusercontent.com/JaxMobile/data_mining/master/datasets/buyProfiles.csv'

A = pd.read_csv(table_A_link)
B = pd.read_csv(table_B_link)
print(A.head())

ws = sm.WhitespaceTokenizer(return_set=True)

output_pairs_name = ssj.jaccard_join(A, B, 'id', 'id', 'name', 'name', ws, 0.3, l_out_attrs=['name'], r_out_attrs=['name'])
print(output_pairs_name.loc[output_pairs_name['_sim_score'] >= 0.5])

output_pairs_des = ssj.jaccard_join(A, B, 'id', 'id', 'description', 'description', ws, 0.3, l_out_attrs=['name'], r_out_attrs=['name'])

print(output_pairs_name)