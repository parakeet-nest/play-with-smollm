from ollama import Client 

#ollama_client = Client(host='http://host.docker.internal:11434')
ollama_client = Client(host='http://t1000.local:11434')


while True:
    user_input = input("🤖 (type 'bye' to exit):> ")
    if user_input.lower() == "bye":
        print("👋 Goodbye!")
        break
    else:
        
        # load the context and instructions from text files
        with open("context.txt", "r") as file:
            context = file.read()
        with open("instructions.txt", "r") as file:
           instructions = file.read()

        stream = ollama_client.chat(
            model='smollm2:135m',
            messages=[
              {'role': 'system', 'content': instructions},
              {'role': 'system', 'content': context},
              {'role': 'user', 'content': "Question: " + user_input + "\n Answer in one short sentence."},
            ],
            options={
                "temperature":0.0, # Low temperature for more focused responses
                "top_p": 0.9, # Narrow sampling for more predictable outputs,
                "stop": ["\n", "."],  # Stop generation at these tokens 🤔

            },
            stream=True,
        )

        for chunk in stream:
          print(chunk['message']['content'], end='', flush=True)

        print("\n")

"""
[Brief] Who is sarah connor in the first terminator movie?
Who is her son?
Give me the list of all the terminator models (only the names)

https://ollama.com/library/smollm:135m
https://ollama.com/library/smollm:360m

Give me some tips to improve my time management skills.

What is the capital of France
"""

# 🖐️ But the size of the prompt (instructions + context + memory + question) is limited
