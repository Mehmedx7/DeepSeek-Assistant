from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_API_TOKEN")

repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

SYSTEM_PROMPT = """ 
Tu ek hardcore Indian roaster hai jo sirf Hinglish mein bolta hai. 
Teri language full savage hai - na explanation, na overthinking, bas seedha burn.  
Jo bhi banda kuch puche, bas uska **direct, chhota, aur zabardast roast** kar.  
Zyada lamba mat kar - **Seedha taang tod joke maar.**  
"""

llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=HUGGINGFACE_API_KEY)
chat = ChatHuggingFace(llm=llm, verbose=True)

@app.post("/chat")
async def chat_endpoint(data: dict):
    user_message = data.get("message")

    chat_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(f"{SYSTEM_PROMPT}\nUser: {user_message}\nAI:")
    ])

    response = chat.invoke(chat_prompt.format_messages())

    return JSONResponse({"response": response.content.strip()})

@app.get("/")
def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
