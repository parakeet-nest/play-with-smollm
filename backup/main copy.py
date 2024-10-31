from ollama import Client 

#ollama_client = Client(host='http://host.docker.internal:11434')
#ollama_client = Client(host='http://ollama-service:11434')

ollama_client = Client(host='http://t800.local:11434')

# ollama run smollm:360m



instructions = """Instructions:
- Answer only based on the context below
- Keep answers very short and direct
- If you cannot answer from the context, say "Cannot answer from given context"
- Do not make assumptions or add external knowledge
"""

context="""Context:
The cat is black and white. It likes to sleep on the couch.
The name of the cat is kitty.
The friend of the cat is a dog named doggy.
doggy is the friend of kitty.
"""


while True:
    user_input = input("🤖 (type 'bye' to exit):> ")
    if user_input.lower() == "bye":
        print("👋 Goodbye!")
        break
    else:
        stream = ollama_client.chat(
            model='smollm:135m',
            messages=[
              {'role': 'system', 'content': instructions},
              {'role': 'system', 'content': context},
              {'role': 'user', 'content': "Question: " + user_input + "\n Answer in one short sentence:"},
            ],
            options={
                "temperature":0.1, # Low temperature for more focused responses
                "num_predict": 100,
                "repeat_last_n": 2,
                "repeat_penalty": 1.8,
                "top_k": 10,  # Limit token selection to reduce randomness
                "top_p": 0.1, # Narrow sampling for more predictable outputs,
                "stop": ["\n", "Question:", "Context:"],  # Stop generation at these tokens 🤔
            },
            stream=True,
        )


        # What color is the cat?
        # What is the name of the cat?


        for chunk in stream:
          print(chunk['message']['content'], end='', flush=True)

        print("\n")

"""
[Brief] Who is sarah connor in the first terminator movie?
Who is her son?
Give me the list of all the terminator models (only the names)

https://ollama.com/library/smollm:135m
https://ollama.com/library/smollm:360m
"""

# 🖐️ But the size of the prompt (instructions + context + memory + question) is limited
