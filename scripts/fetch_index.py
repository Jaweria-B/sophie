import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize OpenAI embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)

# Create a retriever that fetches the top 3 relevant documents
vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define the RAG search prompt template
RAG_SEARCH_PROMPT_TEMPLATE = """
Using the following pieces of retrieved context, answer the question comprehensively and concisely.
Ensure your response fully addresses the question based on the given context.
If you are unable to determine the answer from the provided context, state 'I don't know.'
Question: {question}
Context: {context}
"""

prompt = PromptTemplate.from_template(RAG_SEARCH_PROMPT_TEMPLATE)

def answer_query(question: str) -> str:
    # Retrieve relevant documents
    docs = vectorstore_retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    # Format the prompt with the question and retrieved context
    prompt_text = prompt.format(question=question, context=context)
    messages = [
         {"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": prompt_text},
    ]
    # Call the OpenAI Chat API
    response = openai.ChatCompletion.create(
         # Old
         # model="gpt-4",
         # New
         model="gpt-4o-mini",
         messages=messages,
         temperature=0.1,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer = answer_query(question)
    print("Answer:", answer)
