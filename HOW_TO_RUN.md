# Atrium
## Senior Design Project 
#### University of Pennsylvania

Current Steps for this iteration (in root of project):

1. create python virtual environment: `python -m venv venv` and then start it `source venv/bin/activate`
2. pip install requirements.txt
3. create a .env file and put in an open api key under OPENAI_API_KEY=...
4. run `python models/rag/build_vectorstore.py`
5. run `python models/rag/query_rag.py`
6. will need database credentials to pull from database in .env file too

For running backend + frontend:
1. run `cd backend`
2. run `uvicorn main:app --reload --port 8000`
3. open new terminal `cd frontend`
4. run `npm install` and `npm run dev`
