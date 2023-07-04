import pandas as pd

df = pd.read_excel('/home/nikolas/PycharmProjects/easy_ner_stream01/venv/datatable/input_korr.xlsx')

print(df.columns)
df4 = df[['ИД','Рег. номер','Дата и время регистрации','Корреспондент/адресат','№ документа','Дата документа']]

print(df4.head())
df4.to_csv('/home/nikolas/PycharmProjects/easy_ner_stream01/venv/datatable/input_korr_stat.csv',index=False)