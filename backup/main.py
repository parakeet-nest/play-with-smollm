from ollama import Client 


ollama_client = Client(host='http://host.docker.internal:11434')
#ollama_client = Client(host='http://ollama-service:11434')

instructions = """You are a useful AI agent, your name is Bob. 
Make only short answers."""

# qwen2.5:0.5b 397 MB

while True:
    user_input = input("🤖 (type 'bye' to exit):> ")
    if user_input.lower() == "bye":
        print("👋 Goodbye!")
        break
    else:
        stream = ollama_client.chat(
            model='qwen2.5:0.5b',
            messages=[
              {'role': 'system', 'content': instructions},
              {'role': 'user', 'content': user_input},
            ],
            options={"temperature":0.0},
            stream=True,
        )

        for chunk in stream:
          print(chunk['message']['content'], end='', flush=True)

        print("\n")
