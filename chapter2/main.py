# Step 1: imports

import os
from pathlib import Path
from dotenv import load_dotenv

# loading the variables from the env.txt file to the environment
current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / ".env"

_ = load_dotenv(dotenv_path=env_path, override=True)

from langchain_community.document_loaders import WebBaseLoader
import bs4
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Step 2: OpenAI connection

# setting the OPEN_API_KEY to the library
openai.api_key = os.environ["OPENAI_API_KEY"]

# Step 3: Indexing using web loading and crawling

'''using the WebBaseLoader class from the langchain_community document_loaders module to load web pages as documents.
1. Makes HTTP requests to the specified URLs to fetch the web pages
2. Parses the HTML content of the web pages using BeautifulSoup, considering only the elements specified by the parse_only parameter
3. Extracts the relevant text content from the parsed HTML elements
4. Creates Document objects for each web page that contain the extracted text content, along with metadata such as the source URL

The resulting Document objects are stored in the docs variable
'''

loader = WebBaseLoader(
		web_paths=("https://kbourne.github.io/chapter1.html",),
		bs_kwargs=dict(
			parse_only=bs4.SoupStrainer(
				class_=("post-content","post-title","post-header")
			)
		),
	)

docs = loader.load()

# Step 4: splitting

text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=1000, # Limits each chunk to approximately 1,000 characters.
			chunk_overlap=200, # Includes 200 characters of overlap between consecutive chunks to help preserve context at boundaries.
			length_function=len, # Uses Python’s built-in len() function to measure chunk size.
			is_separator_regex=False, #Treats separators as literal strings rather than regular expressions.
		)

splits = text_splitter.split_documents(docs)

#print(splits)

# Step 5: embbeding and indexing the chunks

# creat the Chroma vector store with the Chroma.from_documents method, which is called to create a Chroma vector store from the split documents.
'''
Internally, the method is doing:
1. It iterates over each Document object in the splits list
2. For each Document object, it uses the provided OpenAIEmbeddings instance to generate an embedding vector
3. It stores the document text and its corresponding embedding vector in the Chroma vector database
'''
vectorstore = Chroma.from_documents(
		documents=splits,
		embedding=OpenAIEmbeddings())

# The retriever is an object that provides a convenient interface for performing these similarity searches and retrieving the relevant documents from the vector database based on those searches.
retriever = vectorstore.as_retriever()

# querying again the vector database

#query = "How does RAG compare with fine-tuning?"
#relevant_docs = retriever.invoke(query)
#print(relevant_docs)

# step 6 prompt templates from LangChain Hub

'''
LangChain Hub is a collection of pre-built components and templates that can be easily integrated into LangChain applications.
It provides a centralized repository for sharing and discovering reusable components, such as prompts, agents, and utilities.
'''

client = Client()
prompt = client.pull_prompt("jclemens24/rag-prompt",dangerously_pull_public_prompt=True)
#print(prompt)

# Step 7: Formatting a function so that it matches the next step's input


"""
a generator expression (doc.page_content for doc in docs) is used to extract the page_content attribute from each document object.
The page_content attribute represents the text content of each document.

The purpose of this function is to format the output of the retriever into the string format that
it will need to be in for the next step in the chain, after the retriever step.
"""
def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)


# Step 8: Defining your LLM

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Step 9: Setting up a LangChain chain using LangChain Expression Language (LCEL)

rag_chain = (
		{
			"context":retriever | format_docs,
			"question":RunnablePassthrough()
		}
		| prompt
		| llm
		| StrOutputParser())



# step 10: submitting a quiestion for RAG

answer = rag_chain.invoke("What are the advantages of using RAG?")
print(answer)

