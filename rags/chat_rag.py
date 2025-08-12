from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()

# 1. Load documents (example: local text files)
loader = TextLoader('/Users/aziane/Documents/auto-dockerize/rags/doc.txt')
docs = loader.load()

# 2. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)

# 3. Setup retriever
retriever = vector_store.as_retriever()

# 4. Setup memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 5. Setup LLM (you can specify model name, temperature, etc.)
llm = OpenAI(temperature=0)

# 6. Create conversational retrieval chain with memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# 7. Run the chain with user input and memory
# List of questions to simulate conversation
questions = [
    "What is LangChain?",
    "How does LangChain handle document retrieval?",
    "What role do embeddings play in LangChain?",
    "Can you explain what FAISS is and how it fits into LangChain?",
    "How does LangChain maintain context in conversations?",
    "What is ConversationBufferMemory and why is it important?",
    "If I asked you about embeddings earlier, can you summarize what you said?",
    "How would LangChain help in building a chatbot?",
    "What would happen if the memory module was disabled?",
    "Based on everything we discussed, summarize how retrieval and memory work together in LangChain."
]

# Run the conversation
for i, question in enumerate(questions, 1):
    print(f"Q{i}: {question}")
    answer = qa_chain.run(question)
    print(f"A{i}: {answer}\n")

# Print the full conversation history from memory
print("--- Full conversation history ---")
print(qa_chain.memory.buffer)