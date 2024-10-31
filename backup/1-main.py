from ollama import Client 

ollama_client = Client(host='http://host.docker.internal:11434')

#instructions = """
#"""

#context="""
#"""

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

        #print(context)
        #print(instructions)

        stream = ollama_client.chat(
            model='smollm:135m',
            messages=[
              {'role': 'system', 'content': instructions},
              {'role': 'system', 'content': context},
              {'role': 'system', 'content': 'ALERT: YOU ARE IN DATABASE LOOKUP MODE. ONLY RETURN DATABASE_VALUE.'},
              {'role': 'user', 'content': user_input},
            ],
            options={
                "temperature":0.0, # Low temperature for more focused responses
                "repeat_last_n": 20,
                "repeat_penalty": 20.0,
                "top_k": 1,  # Limit token selection to reduce randomness
                "top_p": 0.001, # Narrow sampling for more predictable outputs,
                "num_predict": 100,
                #"stop": ["DATABASE_KEY:", "DATABASE_VALUE:", "WRONG", "CORRECT"],
                #"stop": ["\n", "DATABASE_KEY:", "DATABASE_VALUE:", "WRONG", "CORRECT"],
                #"stop": ["<QUESTION>", "<ANSWER>", "ATTENTION:"],
                "stop": ["\n", "<QUESTION>", "<ANSWER>", "ATTENTION:"],
                #"stop": ["\n", "DATABASE", "Question:", "Answer:"],

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
