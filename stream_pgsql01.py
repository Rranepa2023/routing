import datetime

import cv2
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import displayPDF
import tempfile
import os
import re
import json
import easyocr
import shtamp_detect03
import num_recon01
import spacy_kor01
import pdfdatatime
#import classPM
import requests

from xgboost import XGBClassifier
from pdf2image import convert_from_path
from postgres import PGInstance
from info_from_pdf import get_info_from_pdf
from summarizer import preprocess_text, summarizer

st.set_page_config(layout="wide")
reader1 = easyocr.Reader(['en','ru'])
with open("/src/vocab_regex_corr.json") as filik:
    correspondent_regex_and_its_translation_doct = json.loads(filik.read())

path_table = '/src/input_korr.csv'
df3 = pd.read_csv(path_table)

@st.cache_resource
def load_pg_instance():
    pg_instance = PGInstance()
    return pg_instance


st.markdown('''<h1 style='text-align:center;color:#F64A46;'>Сервис маршрутизации входящей корреспонденции</h1>''',unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; color: black;'>Порядок использования сервиса</h2>",unsafe_allow_html=True)
col_left,col_cen,col_right = st.columns(3)
with col_cen:
    st.image('/src/image_routing_2.png',use_column_width='auto')

st.markdown('''<h2 style='text-align: left; color:black;'>Этап 1: загрузите письмо</h2>''',unsafe_allow_html=True)

uploaded_file = st.file_uploader('Загрузите Ваше письмо для распознавания (в pdf-формате)')
if uploaded_file is not None:
    uploaded_file.name = uploaded_file.name[:30]  # обрежем имя файла до 30 символов
    temp_dir = tempfile.mkdtemp('tgt', 'dfd', '/src/temp_files')  # создадим временную папку
    path1 = os.path.join(temp_dir, uploaded_file.name)
    with open(path1, "wb") as fll:
        fll.write(uploaded_file.getvalue())
    #is_button_on = False

but_on = st.button("Распознать")
if but_on:
    print(path1)
    pages1 = convert_from_path(path1)
    bounds3 = []
    pages2 = convert_from_path(path1)
    pages2[0].save('first_page.jpeg','jpeg')

    for i in range(len(pages1)):
        pages1[i].save('image_to_text_'+str(i)+'.jpg','jpeg')
        image2 =  open('image_to_text_'+str(i)+'.jpg','rb')
        f2 = image2.read()
        image2.close()
                
        
        file_bytes2 = np.asarray(bytearray(f2), dtype=np.uint8)
        bytearray_img2 = cv2.imdecode(file_bytes2,1)
        bounds2 = reader1.readtext(bytearray_img2,detail=0,adjust_contrast=0.8)
        bounds2 = ' '.join(bounds2)
        bounds3.append(bounds2)
    #начало функции нахождения штампа и номера документа
    #image_black,shtamp1,num_doc1,res_doc1 = shtamp_detect03.shtamp_det('first_page.jpeg')
    #конец функции
    #pages1[0].save()
    #bounds3 = []
    #for i in
    st.markdown('''<h2 style='text-align: left;color:black;'>Этап 2: выделение текста письма</h2>''',unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
            displayPDF.ipdfViewer(path1)
    with col2:
            bounds3 = ' '.join(bounds3)
            st.text_area('Текст для анализа',bounds3,height=700)
            st.download_button('Выгрузить текст',bounds3)
    #--------------------------------Здесь код
    temp_file_img = 'image_to_text_0.jpg'
    image_black, shtamp1, num_doc1,res_doc1 = shtamp_detect03.shtamp_det(temp_file_img)   #Дата РАНХиГС номер РАНХиГС
    reg_number = res_doc1
    #reg_number = ['6464574']                          #Регистрационный номер РАНХиГС
    #creation_file_date = datetime.date(2000, 1, 1)    #Дата регистрации в РАНХиГС
    creation_file_date = pdfdatatime.findpdf_date(path1)   #Дата регистрации в РАНХиГС
    doc_number = num_recon01.reg_num_stroka(bounds3)  #Номер документа от корреспондента
    document_file_date = datetime.date(2000,1,1)      #Дата документа от корреспондента

    st.markdown('''<h2 style='text-align: left; color: black;'>Этап 3: найденные атрибуты письма</h2>  <h6 style='text-align: left'><span style='color: black;'>(возможна ручная корректировка)</span></h6>''', unsafe_allow_html=True)
    neiro_dat, user_dat = st.columns(2)
    with neiro_dat:
            ranepa_num = st.text_input('Рег.номер: ', reg_number)
            ranepa_dat = st.date_input('Дата регистрации: ', creation_file_date)
            korres_num = st.text_input('Номер документа: ', doc_number)
            korres_dat = st.date_input('Дата создания документа: ', document_file_date)
    with user_dat:
            ranepa_n_user = st.text_input('Рег.номер (пользователь): ',reg_number)
            ranepa_d_user = st.date_input('Дата регистрации (пользователь): ',creation_file_date)
            korres_n_user = st.text_input('Номер документа (пользователь): ', doc_number)
            korres_d_user = st.date_input('Дата создания документа (пользователь): ', document_file_date)
            
    
    #----------Здесь код по выделению корреспондента начинается
    correspondents_list = []
    for pattern in list(correspondent_regex_and_its_translation_doct.keys()):
        f = re.search(pattern, ''.join(''.join(bounds3).lower().split())[:500])
        if f != None:
            correspondent = correspondent_regex_and_its_translation_doct[pattern]
            correspondents_list.append(correspondent)
    correspondents_list = list(set(correspondents_list))
    if len(correspondents_list)==0:
        correspondents_list = spacy_kor01.find_ner_spacy(bounds3)
    #----------Здесь код по выделению корреспондента заканчивается
    #corresp_df = correspondents_list
    neir_kor = st.text_input('Корреспондент по мнению нейронной сети', correspondents_list)
    user_kor = st.selectbox('Корреспондент по мнению пользователя', df3['Корреспондент/адресат'])

    input_df = pd.DataFrame([[ranepa_num,ranepa_dat,korres_num,korres_dat,ranepa_n_user,ranepa_d_user,korres_n_user,korres_d_user]],
                            columns=['Рег.номер','Дата регистрации','Номер документа','Дата создания документа',
                                     'Рег.номер (пользователь)','Дата регистрации (пользователь)','Номер документа (пользователь)','Дата создания документа (пользователь)'])
    
    pg_instance = load_pg_instance()
    
#if approve:
#    st.dataframe(input_df)
#    sql = """INSERT INTO attributes1 (ranepa_num,ranepa_dat,korres_num,korres_dat,ranepa_n_user,ranepa_d_user,korres_n_user,korres_d_user) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""

#    pg_instance.cursor.execute(sql, (input_data['Рег.номер'],
    query = get_info_from_pdf(path1)
    query_for_summariz = preprocess_text(query,list_values=False)
    #---------Здесь код по суммаризации начинается-------
    st.markdown('''<h2 style='text-align: left; color: black;'>Этап 4: краткое содержание письма</h2>''', unsafe_allow_html=True)
    #url = "http://83.143.66.61:5001/generate_summary" 
    url = "http://83.143.66.61:27370/generate_summary"
    #---------Подготовьте данные как JSON
    data = {"text_to_sum": bounds3+'.На русском.'}
    #---------Отправить POST запрос на Flask сервис
    response = requests.post(url, json=data)
    #---------Проверить был ли запрос успешным (код статуса 200)
    if response.status_code == 200:
        #------------------Получить суммаризацию из запроса JSON
        predicted_summary = response.json()['predict_summary']
        st.write(predicted_summary)
    else:
        st.write("Error:", response.text)
    #---------Здесь код по суммаризации заканчивается----

    #---------Здесь код по определению адресата---------------

    w2v_model = pickle.load(open("/src/address/w2v_model.pkl", "rb"))
    encoder = pickle.load(open("/src/address/encoder.pkl", "rb"))
    classes = pickle.load(open("/src/address/classes.pkl", "rb"))
    performers_adress = pickle.load(open("/src/address/performers_adress.pkl", "rb"))

    loaded_model = XGBClassifier()
    loaded_model.load_model('/src/address/model.json')

    def vectorize(sentence):
        words_vecs = [w2v_model.wv[word] for word in sentence if word in w2v_model.wv]
        if len(words_vecs) == 0:
            return np.zeros(300)
        words_vecs = np.array(words_vecs)
        return words_vecs.mean(axis=0)

    query_for_classification = preprocess_text(query,list_values=True)
    letter = np.array([vectorize(query_for_classification)])
    pred = classes[loaded_model.predict(letter)[0]]

    performers = encoder.inverse_transform(pred)
    
    #---------Здесь код по отображению тематики письма и адресата-------------------
    with open("/src/dictionary_adress_role.json") as file_d:
        dict_adresat_with_roles = json.loads(file_d.read())
    st.write(f"Тематика письма: {dict_adresat_with_roles[','.join(performers)]}")
    
    st.markdown('''<h2 style='text-align: left; color: black;'>Этап 5: определение получателя/исполнителя</h2>''', unsafe_allow_html=True)
    st.write("Адресат письма")
    for i in performers:
        try:
            st.write(f"{i}: {performers_adress[i]}")
    
        except:
            pass

    st.dataframe(input_df)
    #approve = st.button("Подтвердить")
    #if approve:
    input_data = {"Рег.номер":ranepa_num,"Дата регистрации": ranepa_dat, "Номер документа": korres_num, "Дата создания документа":korres_dat,"Рег.номер (пользователь)": ranepa_n_user,
                      "Дата регистрации (пользователь)": ranepa_d_user, "Номер документа (пользователь)":korres_n_user,"Дата создания документа (пользователь)":korres_d_user}
    sql = """INSERT INTO attributes1 (ranepa_num,ranepa_dat,korres_num,korres_dat,ranepa_n_user,ranepa_d_user,korres_n_user,korres_d_user) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
    pg_instance.cursor.execute(sql,(input_data['Рег.номер'],
                                        input_data['Дата регистрации'],
                                        input_data['Номер документа'],
                                        input_data['Дата создания документа'],
                                        input_data['Рег.номер (пользователь)'],
                                        input_data['Дата регистрации (пользователь)'],
                                        input_data['Номер документа (пользователь)'],
                                        input_data['Дата создания документа (пользователь)']))
