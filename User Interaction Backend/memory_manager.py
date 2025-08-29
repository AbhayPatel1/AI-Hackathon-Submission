# import os
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.memory import VectorStoreRetrieverMemory

# # Directory to store memory-based vector embeddings
# CHROMA_MEMORY_DIR = os.getenv("CHROMA_MEMORY_DIR", "memory_index")

# # Create and return a VectorStoreRetrieverMemory object
# def get_memory():
#     embedding_model = OpenAIEmbeddings(
#         model="text-embedding-3-small",
#         openai_api_key=os.getenv("OPENAI_API_KEY")
#     )

#     vectorstore = Chroma(
#         collection_name="chat_memory",
#         embedding_function=embedding_model,
#         persist_directory=CHROMA_MEMORY_DIR,
#     )

#     retriever = vectorstore.as_retriever()
    
#     memory = VectorStoreRetrieverMemory(
#         retriever=retriever,
#         memory_key="history",
#         input_key="input",  # ✅ This line fixes the crash
#         return_messages=True
#     )

#     return memory

# # Singleton memory instance
# memory = get_memory()