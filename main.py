import os
from fastapi.responses import HTMLResponse
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings

load_dotenv()
def configure_app():
    app = FastAPI()
    return app
app = configure_app()

OpenAI = os.getenv("OPENAI_API_KEY")

print(f"OPENAI_API_KEY: {OpenAI}")

def setup_chain():
    file = 'Mental_Health_FAQ.csv'
    #TODO template
    template = """
        {context}{question}
        """
    
    embeddings = OpenAIEmbeddings()
    loader = CSVLoader(file_path=file, encoding='UTF-8')
    docs = loader.load()
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    chain_type_kwargs = {"prompt": prompt}

    llm = ChatOpenAI(
        temperature=0.5,
        api_key = OpenAI
        )
    
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        chain_type_kwargs=chain_type_kwargs,
        verbose = True
    )
    return chain

agent = setup_chain()

@app.get("/prompt", response_class=HTMLResponse)
async def get_prompt():
    return """
    <form method="post">
        <input type="text" name="prompt" />
        <input type="submit" />
    </form>
    """

@app.post("/prompt")
async def process_prompt(prompt: str = Form(...)):
        response = agent.run(prompt)
        return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)