from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field

class News(BaseModel):
    stock: str = Field(description="stock symbol")
    date: str = Field(description="date of the news")
    value: float = Field(description="significant acumulated return value of the stock")
    title: str = Field(description="title of the news")
    url: str = Field(description="url of the news")
    source: str = Field(description="source of the news")
    explanation: str = Field(description="explanation of the news relevance to the stock")

class NewsSet(BaseModel):
    news: List[News] = Field(description="list of news")