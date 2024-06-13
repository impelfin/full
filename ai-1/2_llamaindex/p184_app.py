import os
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import GPTVectorStoreIndex, ServiceContext, LangchainEmbedding, LLMPredictor
from llama_index import SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI

documents = SimpleDirectoryReader('data').load_data()

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
    model_name="bongsoo/moco-sentencedistilbertV2.1"
))
print("Embedding Model initialized successfully")

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
print("LLM Predictor initialized successfully")

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
print("Service Context initialized successfully")

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
print("Index initialized successfully")

query_engine = index.as_query_engine()
print("Query Engine initialized successfully")

query = "미코의 소꿉친구 이름은?"
response = query_engine.query(query)
print(f"Query: {query}", end="\n\n")
print(f"Response: {response}", end="\n\n")
print(f'source_nodes: {response.source_nodes}', end="\n\n")
