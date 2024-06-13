import openai

response = openai.Moderation.create(
    input='I kill you!'
)

print(response)
