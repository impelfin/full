from llama_index import GPTVectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index import PromptHelper

documents = SimpleDirectoryReader('data').load_data()

prompt_helper = PromptHelper(
    max_input_size=4096, # LLM 입력의 최대 토큰 수
    num_output=256, # LLM 출력의 최대 토큰 수
    max_chunk_overlap=20, # 청크 오버랩의 최대 토큰 개수
    )

service_context = ServiceContext.from_defaults(
    prompt_helper=prompt_helper
)

index = GPTVectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
)

print('index: ', index)
