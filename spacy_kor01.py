import spacy
import re

def find_ner_spacy(stroka_in):
    text1 = re.sub(r'[,"-?:!;\/]', ' ', stroka_in)
    nlp1 = spacy.load('ru_core_news_lg')
    doc1 = nlp1(text1)
    for ent in doc1.ents:
        if ent.label_ == 'ORG':
            return ent.text

#print(find_ner_spacy('МИНИСТЕРСТВО ИНОСТРАННЫХ ДЕЛ РОССИЙСКОЙ ФЕДЕРАЦИИ РЕКТОРУ (МИД РОССИИ) РОССИЙСКОЙ АКАДЕМИИ НАРОДНОГО Смоленская-Сенная площадь; дом 32/34, ХОЗЯЙСТВА И ГОСУДАРСТВЕННОЙ'))
#print(find_ner_spacy('МИНИСТЕРСТВО ИНОСТРАННЫХ ДЕЛ РОССИЙСКОЙ ФЕДЕРАЦИИ РЕКТОРУ (МИД РОССИИ) РОССИЙСКОЙ АКАДЕМИИ НАРОДНОГО Смоленская-Сенная площадь; дом 32/34, ХОЗЯЙСТВА И ГОСУДАРСТВЕННОЙ'))

