import os
import datetime

import numpy as np
import pandas as pd
import pdfdatatime
import spacy_kor01

df5 = pd.read_csv('/home/nikolas/PycharmProjects/easy_ner_stream01/venv/datatable/input_korr_stat.csv')
strok1 = df5.shape[0]                                                                                                   #Определим сколько всего строк в таблице
print('Количество строк: ', strok1)                                                                                     #Выведем количество строк
path_pdf = '/home/nikolas/PycharmProjects/easy_ner_stream01/venv/pdfinput/'                                             #Определим путь ко всем документам PDF
nachalo,konec = 2,40

all_files_pdf = os.listdir(path_pdf)
all_files_pdf = all_files_pdf[nachalo:konec]

df5 = df5.reindex(columns = df5.columns. tolist () +
                            ['Рег. номер_1','Дата и время регистрации_1',
                             'Корреспондент/адресат_1','№ документа_1','Дата документа_1'])                         #Добавим пустых колонок

for i in all_files_pdf:
    file_find = path_pdf + i
    creation_file_date = pdfdatatime.findpdf_date(file_find)
    in_table = int(i[:-6])
    print(df5[df5['ИД']==in_table])
    df7 = df5[df5['ИД']==in_table]
    meta_date = str(creation_file_date.date())
    real_date = df7['Дата и время регистрации'].values[0][:10]                      #Берем из колонки только значение даты (это 10 символов)
    print(meta_date , 'Эталонное значение: ', real_date)

    spac_korr = spacy_kor01.find_ner_spacy(stroka_pism)
    real_korr = df7['Корреспондент/адресат'].values[0]
    if meta_date == real_date:
        df5.loc[df5['ИД']==in_table, ['Дата и время регистрации_1']] = 1


df5.to_csv('input_korr_stat_new.csv',index=False)


