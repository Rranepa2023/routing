import PyPDF2

def findpdf_date(date_time1):
    with open(date_time1,'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        creationDate = pdf_reader.metadata.creation_date
        return  creationDate
