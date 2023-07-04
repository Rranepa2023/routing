import os
import re
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import easyocr
import boxesdrawer
import pdfdatatime
import docnum_find
import doc_num_korr
import spacy_kor01

from pdf2image import convert_from_path
from datetime import datetime

reader1 = easyocr.Reader(['en','ru'])
path_table = '/home/nikolas/PycharmProjects/easy_ner_stream01/venv/datatable/input_korr.csv'
path_pdf = '/home/nikolas/PycharmProjects/easy_ner_stream01/venv/pdfinput/'
temp_img = '/home/nikolas/PycharmProjects/easy_ner_stream01/venv/tempimage/'
temp_file_img = temp_img + 'image1.jpeg'

df3 = pd.read_csv(path_table)
spisok_pdf = os.listdir(path_pdf)

st.header('Выберите письмо из представленных ниже')
file_name = st.selectbox('Письмо входящее',spisok_pdf)
st.write('Вы выбрали ',file_name)
but_on = st.button('Распознать')
if but_on:
    creation_file_date = pdfdatatime.findpdf_date(path_pdf+file_name)
    #document_file_date = creation_file_date
    pages2 = convert_from_path(path_pdf+file_name)
    pages2[0].save(temp_file_img,'jpeg')
    image1 = open(temp_file_img,'rb')
    f = image1.read()
    image1.close()
    file_bytes = np.asarray(bytearray(f),dtype=np.uint8)
    bytearray_img = cv2.imdecode(file_bytes,1)
    bounds1 = reader1.readtext(bytearray_img, detail=1, adjust_contrast=0.8)
    image2 = boxesdrawer.draw_boxes(temp_file_img,bounds1)
    image2.save('out111.jpg')
    st.image('out111.jpg')
    file_list = open('bounds_list.txt','wt')
    file_list.write(str(bounds1))
    file_list.close()

    stroka_pism = ''
    for i in bounds1:
        stroka_pism = stroka_pism + ' '+ i[1]

    all_file_dates = re.findall(r'\d\d\.\d\d\.\d{4}', stroka_pism)

    if all_file_dates:
        try:
            date_time_obj = datetime.strptime(all_file_dates[0], '%d.%m.%Y')
            document_file_date = date_time_obj.date()
        except ValueError:
            document_file_date = creation_file_date
    else:
        document_file_date = creation_file_date

    #Поиск строки Вход Ng
    reg_number = docnum_find.find_doc_num(stroka_pism)
    #Поиск Номера Документа
    doc_number = doc_num_korr.reg_num_stroka(stroka_pism)
    #Запись строки письма в файл
    str_text = open('pismotext.txt','wt')
    str_text.write(stroka_pism)
    str_text.close()

    number_doc, koresp, data_doc = st.columns(3)

    with number_doc:
        d_num = st.text_input('Номер документа: ', doc_number)
        r_num = st.text_input('Рег. номер: ', reg_number)
    with koresp:
        st.text_input('Корр. (по мнению NER-сети): ',spacy_kor01.find_ner_spacy(stroka_pism))
        k_spa = st.selectbox('Корр. (по мнению пользователя): ',df3['Корреспондент/адресат'])
    with data_doc:
        d_dat = st.date_input('Дата документа: ', document_file_date)
        c_dat = st.date_input('Дата регистрации: ', creation_file_date)





