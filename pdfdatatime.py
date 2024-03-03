import PyPDF2
import datetime

def findpdf_date(date_time1):
    with open(date_time1,'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        try:
            creationDate = pdf_reader.metadata.creation_date
        except AttributeError:
            creationDate = datetime.date(2000, 1, 1)
        return  creationDate
