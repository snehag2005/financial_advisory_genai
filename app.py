# {{{ imports

from fin_data_qa import analyze_financial_data
from vector_db_qa import query_vector_db

from langchain_openai import AzureChatOpenAI
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

import yfinance as yf
from pydantic import BaseModel, Field
from typing import List

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
# }}}

# {{{ class: financial data analysis

class FinancialDataAnalysisInput(BaseModel):
    query: str = Field(..., description='Query string for financial data analysis')

class FinancialDataAnalysisTool(BaseTool):
    name = 'analyze_financial_data'
    description = """Useful to analyze user's personal finances like spending habits, goals, risk profile, etc."""

    def _run(self, query: str):
        with st.spinner('Looking up your financial data...'):
            response = analyze_financial_data(query)
            st.success('✅ Analyzed your financial data')
        return response

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = FinancialDataAnalysisInput

# }}}

# {{{ class: vector db recommendation

class RecommendProductsInput(BaseModel):
    query: str = Field(..., description='Query for Bank of Baroda product recommendations')

class RecommendProductsTool(BaseTool):
    name = 'query_vector_db'
    description = "Useful for recommending BoB products or answering BoB-specific queries"

    def _run(self, query: str):
        with st.spinner('Searching BoB database...'):
            response = query_vector_db(query)
            st.success('✅ Found relevant info from BoB database')
        return response

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = RecommendProductsInput

# }}}

# {{{ stock price lookup

def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    try:
        data = ticker.history(period='1d')
        return round(data['Close'][0], 2)
    except:
        return 0

class StockPriceLookupInput(BaseModel):
    symbol: str = Field(..., description='Stock ticker symbol')

class StockPriceLookupTool(BaseTool):
    name = 'get_stock_price'
    description = 'Look up current price of a stock by its ticker symbol'

    def _run(self, symbol: str):
        with st.spinner('Getting stock price...'):
            price = get_stock_price(symbol)
            st.success('✅ Fetched stock price')
        return price

    def _arun(self, symbol: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceLookupInput

# }}}

# {{{ google search tool

tavily_tool = TavilySearchResults(max_results=20)

def search_investment_options(query):
    return tavily_tool.invoke(query)

class SearchInvestmentOptionsInput(BaseModel):
    search_query: str = Field(..., description="Query for Google investment search")

class SearchInvestmentOptionsTool(BaseTool):
    name = 'search_investment_options'
    description = """Suggest investment options or stocks based on user profile. Focus on India-specific info."""

    def _run(self, search_query: str):
        with st.spinner('Searching investment options...'):
            results = search_investment_options(search_query)
            st.success('✅ Fetched investment suggestions')
        return results

    def _arun(self, search_query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = SearchInvestmentOptionsInput

# }}}

# {{{ agent setup

tools = [
    FinancialDataAnalysisTool(),
    RecommendProductsTool(),
    StockPriceLookupTool(),
    SearchInvestmentOptionsTool(),
]

llm = AzureChatOpenAI(
    temperature=0,
    openai_api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

open_ai_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# }}}

# {{{ streamlit app

st.set_page_config(page_title='BoB Bot', page_icon='🏆')
st.title('🏆 BoB Bot - Financial & Investment Advisor')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_response(query, chat_history):
    template = """You are an expert financial advisor at Bank of Baroda.
Provide answers based on user queries, using the tools available. Use bullet points if needed.

Chat history:
{chat_history}

User question:
{user_question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | open_ai_agent

    with st.spinner('Thinking...'):
        result = chain.invoke({
            "chat_history": chat_history,
            "user_question": query
        })
    return result['output']

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message("Human" if isinstance(message, HumanMessage) else "AI"):
        st.markdown(message.content)

# Suggested questions
suggestions = [
    "Analyze my spending habits and share ways to optimize",
    "Based on my goals, recommend some BoB products",
    "As per my risk profile, can you suggest some investment options?",
]
st.markdown("**Try one of these:**")
cols = st.columns(len(suggestions))
for i, q in enumerate(suggestions):
    if cols[i].button(q):
        st.session_state.chat_history.append(HumanMessage(q))
        with st.chat_message("Human"):
            st.markdown(q)
        with st.chat_message("AI"):
            response = get_response(q, st.session_state.chat_history)
            st.markdown(response)
        st.session_state.chat_history.append(AIMessage(response))

# Chat input
user_query = st.chat_input("Your question...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(response))

# }}}
