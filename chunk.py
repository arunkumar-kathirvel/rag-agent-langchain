# pip install langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # aim for ~800 chars per chunk
    chunk_overlap=100,     # adjacent chunks share 100 chars (context glue)
)

# Open the PDF
reader = PdfReader("chunking_test_document.pdf")

# Loop through pages and extract text
pdfContent = ""
for page in reader.pages:
    pdfContent += page.extract_text() + "\n"

chunks = splitter.split_text(pdfContent)

print(len(chunks))    # → e.g. 47 chunks ready to embed

for chunk in chunks:
    print(chunk + "\n")