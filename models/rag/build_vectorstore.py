# build_vectorstore.py
import os, pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()  # reads .env with OPENAI_API_KEY=...

# Validate OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in your environment or create a .env file.\n"
        "Example: OPENAI_API_KEY=sk-..."
    )

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to data/philadelphia-pa-1.txt (go up 2 levels from models/rag/ to project root)
data_path = os.path.join(current_dir, "../../data/philadelphia-pa-1.txt")

# Normalize (resolve ..)
data_path = os.path.normpath(data_path)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = [Document(page_content=t) for t in splitter.split_text(text)]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vectorstore to data/faiss_index (relative to project root)
save_path = os.path.join(current_dir, "../../data/faiss_index")
save_path = os.path.normpath(save_path)
vectorstore.save_local(save_path)
