import re

def reg_num_stroka(stroka_in):
    spis_template = ['(\d\d-\d{7})','(\d\d-\d\d-\d{5}/\d\d)','(\d{5}/\d\d)','(\d\d-\d\d-\d\d/\d\d)','(\d{3}-\d\d-\d-\d\d)']
    stroka_all = '(дм-п1)'
    for i in spis_template:
        stroka_all=stroka_all+'|'+i
    stroka_num = re.search(stroka_all,stroka_in,flags=re.IGNORECASE)
    if stroka_num:
        recom1 = stroka_num.group(0)
    else:
        recom1 = '0000000'
    return recom1

