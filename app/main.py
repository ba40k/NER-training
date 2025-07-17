from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from app.ner_pipeline import fetch_text_from_url, extract_products_hf
import os

app = FastAPI()

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class URLRequest(BaseModel):
    url: str

class ProductResponse(BaseModel):
    url: str
    products: List[str]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extract-products/", response_model=ProductResponse)
async def extract_products(url_request: URLRequest):
    try:
        text = fetch_text_from_url(url_request.url)
        products = extract_products_hf(text)
        return {"url": url_request.url, "products": products}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))