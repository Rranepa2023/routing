import easyocr
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

import pymorphy2
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()

import numpy as np
import re




def get_info_from_pdf(pdf: 'str'): 
    """На вход функции подаётся адрес до pdf-документа
    \nНа выходе получается список строк распознанного текста
    \n...а также средняя оценка плохого распознавания текста: чем выше (выше 0.75) avg_bad_text_recognition_score - тем хуже распознан текст """
    
    regular_pattern = re.compile(r'\w\w')


    # ШАГ 1: пробуем распознать "digitally-created PDF":
    pdf_text_list = []
    # ридер из PyPDF2:
    reader = PdfReader(pdf)

    # собираем информацию с каждой страницы pdf-документа...:
    number_of_pages = len(reader.pages)
    for i in range(number_of_pages):
        page = reader.pages[i]
        try:
            text = page.extract_text()        
            if text != '': # если текст на странице НЕ пустой
                text = re.split('/n|\n', text)  # делим построчно, сохраняя структуру документа
                pdf_text_list.extend(text) # добавляем в единый список текста всей pdf
        except:
            pdf_text_list = []

    # если полученный список не пустой...:
    if pdf_text_list != []:
        pdf_text_info = pdf_text_list

        avg_bad_text_recognition_score = 0

    # ШАГ 2: если это не цифровая pdf, а картинка - исползуем EasyOCR:
    else:
        # ридер из EasyOCR:
        reader = easyocr.Reader(['ru', 'en'])  
        image = convert_from_path(pdf)
        image_text_list = []
        for page in range(len(image)):
            page_np_array = np.array(image[page])
            text = reader.readtext(page_np_array, detail = 0, paragraph=True, 
                                #    rotation_info=[90, 180 ,270],
                                #    contrast_ths=0.1,
                                #    adjust_contrast=0.6 # 0.6-плохо, 0.4-плохо 0.5
                                ) 
            image_text_list.extend(text)
        
        pdf_text_info = image_text_list

        # оцениваем качество распознвания EasyOCR:
        text_recognition_score = []
        for string in image_text_list:
            count = 0
            bad = 0
            for word in regular_pattern.findall(string):
                count += 1
                parsed_info = morph.parse(word)[0]
                bad += (not parsed_info.tag.POS or 
                    isinstance(parsed_info.methods_stack[0][0], pymorphy2.units.by_analogy.KnownSuffixAnalyzer.FakeDictionary)) # FakeDictionary - значит, что текст не распознан
            try:
                text_recognition_score.append(bad/count) # по каждой строке добавялем скор-отношение плохо распознанных слов к количеству всех слов
            except:
                text_recognition_score.append(0.5) # если добавить не получилось - добавляем среднее значение 0.5


        # высчитываем средний скор по всем строкам:
        avg_bad_text_recognition_score = sum(text_recognition_score) / len(text_recognition_score)

    return pdf_text_info, avg_bad_text_recognition_score
         
         