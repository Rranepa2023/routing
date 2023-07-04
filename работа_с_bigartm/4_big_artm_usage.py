import artm
from artm import hARTM

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")

import gensim
from gensim import corpora

from pymystem3 import Mystem

import re
from string import punctuation




#----ПРЕДОБРАБАТЫВАЕМ ТЕКСТ до создания vowpal wabbit ФАЙЛА----
def preprocess_text(raw_text: str, mystem=Mystem()) -> list:
  '''
  Функция для предобработки текста перед созданием vowpal wabbit файла
  raw_text - строка на вход
  На выходе получаем text - список отдельных слов
  '''
  
  text = re.sub(r'[\(\)\[\]\{\}\.\,\!\?\:\"\'\_\`\„\“\^\&\№\¡\™\£\¢\∞\§\¶\•\ª\º\≠\“\”\`\…\æ\«\»\>\<\≤\≥\÷\µ\=\~\;\/\’\\xad\*\‘\%\”\\\\꞉\·\'\˚\d+\s|\s\d+\s|\s\d+$][a-zA-Z0-9]*', ' ', raw_text) # удаляем ненужные символы
  text = re.sub(r'[\-\—\–\−\‑️\‒]{2,}', ' ', text)
  lemmas = mystem.lemmatize(text.lower())
  tokens = [token for token in lemmas if token not in russian_stopwords\
            and token != " " \
            and token.strip() not in punctuation\
            and len(token)>1
            ] # будет список с разными пробелами (одинарными и не только), как элементами списка

  text = ' '.join(tokens).split() # будет список чисто из слов        " ".join(tokens) - будет строка

  return text



#----ДЕЛАЕМ КОРПУС train БАТЧЕЙ----
def make_corpus_batches(path_to_vw_file: str, batches_folder_path: str):
  '''
  \npath_to_vw_file - путь к файлу vowpal wabbit, созданному самостоятельно из корпуса текстов для обучения модели
  \nbatches_folder_path - путь к папке для формирования батчей из этого файла vowpal wabbit
  \nРезультат функции - batches_folder_path - папка с сформированными батчами для обучения модели big artm
  \nНа выход функция ничего не отдает
  '''

  batch_vectorizer=artm.BatchVectorizer(data_path=path_to_vw_file,
                                        data_format='vowpal_wabbit',
                                        target_folder=batches_folder_path)
  

#--------СОЗДАЕМ И ОБУЧАЕМ МОДЕЛЬ--------
def model_training(batches_folder_path:str, decorrelator_phi, SparseTheta, SparsePhi, num_topics:int):
  '''
  \nФункция создания и обучения модели BigARTM
  \nbatches_folder_path - путь к папке с батчами. Указывался в функции def make_corpus_batches()
  \ndecorrelator_phi - коэффициент декорреляции
  \nSparseTheta - коэффициент разреженности матрицы Тета (Θ) 
  \nSparsePhi - коэффициент разреженности матрицы Фи (Φ)
  \nnum_topics - количество тем для построения модели   
  '''

  # загружаем имеющиеся batch-и:
  batch_vectorizer = artm.BatchVectorizer(data_path=batches_folder_path,data_format='batches')
  dictionary = artm.Dictionary('dictionary')
  dictionary.gather(data_path=batches_folder_path)

  hier = hARTM()

  level0 = hier.add_level(num_topics=num_topics)
  # level0.class_ids = {'@default_class': 1.0} # без модальностей
  level0.num_processors = 32
  level0.cache_theta = True

  # задаём внутренние метрики
  level0.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=20))
  level0.scores.add(artm.PerplexityScore(name='perplexity_score', dictionary=dictionary))
  level0.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
  level0.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
  level0.scores.add(artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.5))  # от 0.5 и выше

  # задаём регуляризатор декоррелирования:
  level0.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi'))
  level0.regularizers['DecorrelatorPhi'].tau = decorrelator_phi # лучший найденный К декорреляции Φ

  # задаём регуляризатор разреживания матрицы Θ (тем-документов):
  level0.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta'))
  level0.regularizers['SparseTheta'].tau = SparseTheta # лучший найденный К разреженности Θ

  # задаём регуляризатор разреживания матрицы Φ (термов-тем):
  level0.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi'))
  level0.regularizers['SparsePhi'].tau = SparsePhi  # лучший найденный К разреженности Φ

  # инициализируем модель:
  level0.initialize(dictionary=dictionary)

  # обучаем
  level0.fit_offline(batch_vectorizer, num_collection_passes=8)

  return level0


#--------НОВЫЙ ТЕКСТ ПРЕОБРАЗУЕМ В БАТЧИ ДЛЯ МОДЕЛИ--------
def make_query_batch(query: str, path_to_save_query_vw: str, path_to_save_query_batches):
  '''
  \nФункция для перевода нового текста в формат для подачи в модель big artm
  \nquery - новый текст/запрос/оцифрованный текст письма (pdf, изображения и т.д.) - строка
  \nbatch_vectorizer_query - объект типа artm.BatchVectorizer на папку с батчем нового текста
  '''
  if len(query) != 0:
      query_list = preprocess_text(query)  # один список чисто из слов

  qqq = []
  qqq.append(query_list) # список из одного списка слов
  query_dictionary = corpora.Dictionary(list(qqq))
  query_corpus = [query_dictionary.doc2bow(text) for text in qqq]
  word_query_counts = [[(query_dictionary[id], count) for id, count in line] for line in query_corpus] # список из одного списка тюплов

  f = open(path_to_save_query_vw, 'w+')
  for text in word_query_counts:
      z = []
      z = [ (item[0] + ":" + str(int(item[1])) ) for item in text]  # частота упоминания через : от слова - такого вида: ['report:1', 'russi:1']
      f.write((" ").join(z) + '\n')  # запись в мешок слов - такого вида: report:1 russi:1
  f.close()


  batch_vectorizer_query = artm.BatchVectorizer(
                                                data_path = path_to_save_query_vw,
                                                data_format='vowpal_wabbit',
                                                target_folder = path_to_save_query_batches)  # папка с частотной матрицей из batch)

  return batch_vectorizer_query
  

#--------ПОЛУЧАЕМ ТЕМЫ, К КОТОРЫМ ОТНОСИТСЯ НОВЫЙ ТЕКСТ--------
def define_query_into_topics(query:str, level0, named_topics: dict):
  '''
  \nФункция показывает наиболее вероятные темы обученной модели big artm для нового текста
  \nquery - новый текст/запрос/оцифрованный текст письма (pdf, изображения и т.д.) - строка
  \nlevel0 - обученная модель big artm (выход функции model_training())
  \nnamed_topics - словарь, где ключ - номер topic-а из обученной модели, значение - строка-именование этой темы (от асессора)
  '''
  # сделаем из него батч:
  batch_vectorizer_query = make_query_batch(query)

  query_theta = level0.transform(batch_vectorizer=batch_vectorizer_query) # get level-wise vertically stacked Theta matrices for new documents
  query_theta_dict = query_theta.to_dict('dict') # матрица тем-документов по нашему запросу

  # theta = level0.get_theta().transpose() # предыдущая матрица тем-документов, у обученной модели
  # theta_dict = theta.to_dict('dict')

  # словарь для сохранения результата:
  topics_and_prob_for_query = {}

  # для каждого нового текста (query-запроса) отсортируем подходящие темы по убыванию вероятности:
  for doc_id, values in query_theta_dict.items(): # т.к. 1 запрос - тут будет только 1 пара ключ (id документа-запроса) - значение (вероятности тем)
    sorted_values = sorted(values.items(), key=lambda item: item[1], reverse=True)

    # добавляем в результат первые 5 тем, наиболее подходящие к запросу, и их вероятность:
    for i in range(5):
        topics_and_prob_for_query[sorted_values[i][0]] = round(sorted_values[i][1]*100, 2)

  return topics_and_prob_for_query










# query = pdf_text_info # результат функции get_info_from_pdf()

# decorrelator_phi = 15e4
# SparseTheta = -0.40
# SparsePhi = -0.01
# num_topics = 15

# named_topics = {'topic_0': 'Про документы об образовании',
#                 'topic_1': 'Автомобильные штрафы',
#                 'topic_2': 'Судебные письма и взыскания',
#                 'topic_3': 'Про документы об образовании',
#                 'topic_4': 'Проверки и строительство',
#                 'topic_5': 'Требование документов вышестоящими органами',
#                 'topic_6': 'Предложения ,тендеры, конкурсы',
#                 'topic_7': 'Мин.Обороны',
#                 'topic_8': 'Письма о решениях, оценках, экспертизах',
#                 'topic_9': 'Письма об изменениях в бюджетах',
#                 'topic_10': 'Рабочие группы и заседания (?)',
#                 'topic_11': 'Конкурсы, конгрессы, собрания, фестивали',
#                 'topic_12': 'Приглашения к мероприятиям (?)',
#                 'topic_13': 'Запросы/приглашения (?)',
#                 'topic_14': 'Covid-19'
#                 }


# make_corpus_batches('texts_sum_corresp_and_bigrams_vw.txt', 'tsc_bigrams_routing_batches')
# level0 = model_training('tsc_bigrams_routing_batches', decorrelator_phi=decorrelator_phi, SparseTheta=SparseTheta, SparsePhi=SparsePhi, num_topics=num_topics)
# batch_vectorizer_query = make_query_batch(query, 'query_vw.txt', 'query_routing_batches')
# topics_and_prob_for_query = define_query_into_topics(query, level0, named_topics)
#for key in topics_and_prob_for_query.keys():
# print(f'')







