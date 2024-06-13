import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

tokens = enc.encode("안녕. 세상아!!")
print(len(tokens))
print(tokens)

print(enc.decode(tokens))

def data2str(data):
    try:
        return data.decode('utf-8')
    except UnicodeError:
        return data
    
print([data2str(data) for data in enc.decode_tokens_bytes(tokens)])
