import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

tokens = enc.encode("hello world")
print(len(tokens))
print(tokens)

print(enc.decode(tokens))

print(enc.decode_tokens_bytes(tokens))
