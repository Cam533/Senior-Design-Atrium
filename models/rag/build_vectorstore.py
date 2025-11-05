# build_vectorstore.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Add it to .env")

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path_code = os.path.normpath(os.path.join(current_dir, "../../data/philadelphia-pa-1.txt"))
data_path_checklist = os.path.normpath(os.path.join(current_dir, "../../data/Development_Checklist-July-2024.pdf"))

# Load zoning text
with open(data_path_code, "r", encoding="utf-8") as f:
    text_code = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_code = [Document(page_content=t, metadata={"source": "philadelphia-pa-1.txt"}) for t in splitter.split_text(text_code)]

# Load checklist PDF
# TODO: In order to improve accuracy of the checklist, we should tabulate the data so for 
# each requirement listed in the checklist, we can have a list of contacts and links to 
# relevant documents.
loader = PyPDFLoader(data_path_checklist)
pdf_docs = loader.load()
docs_checklist = splitter.split_documents(pdf_docs)
for d in docs_checklist:
    d.metadata["source"] = "Development_Checklist-July-2024.pdf"


# Embed & save
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore_code = FAISS.from_texts(
    [doc.page_content for doc in docs_code],
    embeddings,
    metadatas=[doc.metadata for doc in docs_code]
)

vectorstore_checklist = FAISS.from_texts(
    [doc.page_content for doc in docs_checklist],
    embeddings,
    metadatas=[doc.metadata for doc in docs_checklist]
)

save_path_code = os.path.normpath(os.path.join(current_dir, "../../data/faiss_index_code"))
save_path_checklist = os.path.normpath(os.path.join(current_dir, "../../data/faiss_index_checklist"))
if os.path.exists(save_path_code) and os.path.exists(save_path_checklist):
    import shutil
    shutil.rmtree(save_path_code)
    shutil.rmtree(save_path_checklist)

vectorstore_code.save_local(save_path_code)
vectorstore_checklist.save_local(save_path_checklist)
print(f"Vectorstore built and saved to {save_path_code} and {save_path_checklist}")
