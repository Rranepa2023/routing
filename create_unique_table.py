import pandas as pd

df = pd.read_excel('/home/nikolas/PycharmProjects/easy_ner_stream01/venv/datatable/input_korr.xlsx')

print(df.columns)
print(df['Корреспондент/адресат'].shape)
df2 = df['Корреспондент/адресат'].unique()

print(df2)
print(df2.shape)
df3 = pd.DataFrame(df2, columns = ['Корреспондент/адресат'])
print(df3)
df3.to_csv('/home/nikolas/PycharmProjects/easy_ner_stream01/venv/datatable/input_korr.csv')