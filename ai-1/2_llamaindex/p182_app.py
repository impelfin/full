from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI

documents = SimpleDirectoryReader('data').load_data()
print('documents: ', documents)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

index = GPTVectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)

print('index: ', index)

query_engine = index.as_query_engine()

print('query: ', query_engine.query('미코의 성격은?'))
