import re

def reg_num_stroka(stroka_in):
    stroka_num = re.search('(дм-п1)|(\d\d-\d{7})|(\d\d-\d\d-\d{5}/\d\d)|(\d{5}/\d\d)',stroka_in, flags=re.IGNORECASE)
    if stroka_num:
        recom1 = stroka_num.group(0)
    else:
        recom1 = '0000000'
    return  recom1
