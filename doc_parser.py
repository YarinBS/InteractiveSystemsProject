from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from PyPDF2 import PdfReader


def docx_to_dict(path_to_docx):
    document = Document(path_to_docx)
    section = document.sections[0]
    docx_texts = []

    # Paragraph or table
    for element in section.iter_inner_content():
        if isinstance(element, Paragraph):
            if element.text != '':
                docx_texts.append(element.text)
        elif isinstance(element, Table):
            row_num = 1
            headers_row = []
            for row in element.rows:
                new_row = 'row_' + str(row_num) + ': '
                if row_num == 1:
                    for cell in row.cells:
                        headers_row.append(cell.text)
                        new_row += " " + cell.text
                    row_num += 1
                    docx_texts.append(new_row)
                else:
                    for i, cell in enumerate(row.cells):
                        new_row += " " + headers_row[i] + ": " + cell.text + "; "
                    docx_texts.append(new_row)
                    row_num += 1

    return docx_texts


def txt_text(file_path: str):
    file_paragraphs = []
    with open(file_path, 'r') as file:
        file_content = file.read()

        # Split the content into paragraphs based on double newline characters
        paragraphs = file_content.split('\n\n')

        # Print each paragraph
        for i, paragraph in enumerate(paragraphs, start=1):
            file_paragraphs.append(paragraph)

        return file_paragraphs


def pdf_text(file_path):
    reader = PdfReader(file_path)
    texts = []

    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        texts.append(text.strip())

    return texts


def create_chunks(texts: list, chunk_size: int = 256, overlapping: bool = True, overlapping_size: int = 64):
    complete_text = ' '.join([text for text in texts])
    words = complete_text.split()

    if overlapping:
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlapping_size)]
    else:
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    return chunks
