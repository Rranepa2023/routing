
def find_doc_num(stroka_in):
    index_find = stroka_in.find('Вход')
    reg_number = stroka_in[index_find + 7:index_find + 15]
    return reg_number
