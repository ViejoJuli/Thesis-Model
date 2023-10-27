#!/usr/bin/python3
from PyPDF2 import PdfReader

pdf = PdfReader("user manual.pdf")
print(pdf)
