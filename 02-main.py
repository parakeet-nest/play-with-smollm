import math
from ollama import Client 

def dot_product(vector1, vector2):
    return sum(a * b for a, b in zip(vector1, vector2))

def magnitude(vector):
    return math.sqrt(sum(a * a for a in vector))

def cosine_similarity(vector1, vector2):
    return dot_product(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))

def cosine_distance(vector1, vector2):
    return 1 - cosine_similarity(vector1, vector2)

#ollama_client = Client(host='http://host.docker.internal:11434')
ollama_client = Client(host='http://t800.local:11434')

fact1 = 'Philippe Charrière is a Solution Architect at 🐳 Docker.'
fact2 = 'Palmipod is a purple blob expert with 💜 WASM.'
fact3 = 'KeegOrg is a monster in the 🎲 RPG Game "Chronicle of Swords".'

vec1 = ollama_client.embed(model='all-minilm', input=fact1)
vec2 = ollama_client.embed(model='all-minilm', input=fact2)
vec3 = ollama_client.embed(model='all-minilm', input=fact3)

vectorDb = [
    (vec1['embeddings'][0], fact1),
    (vec2['embeddings'][0], fact2),
    (vec3['embeddings'][0], fact3),
]

print("📚 Vector Database:", vectorDb)

while True:
    user_input = input("🤖 (type 'bye' to exit):> ")
    if user_input.lower() == "bye":
        print("👋 Goodbye!")
        break
    else:
        
        # load the context and instructions from text files
        #with open("context.txt", "r") as file:
        #    context = file.read()

        with open("instructions.txt", "r") as file:
           instructions = file.read()

        embeddings = ollama_client.embed(model='all-minilm', input=user_input)

        # Initialize variables to store the minimum distance and corresponding fact
        min_distance = float('inf')
        closest_fact = None

        # calculate the cosine distance between the user input and the vectors in the database
        # and return the closest match

        for item in vectorDb:
            # Unpacking the tuple
            vector, fact = item
            distance = cosine_distance(embeddings['embeddings'][0], vector)

            print("📝", fact, distance)

            # Update the minimum distance and corresponding fact if a smaller distance is found
            if distance < min_distance:
                min_distance = distance
                closest_fact = fact

        # Print the closest fact and its distance
        print(f"Closest match: {closest_fact} with distance {min_distance}")


        stream = ollama_client.chat(
            model='smollm:135m',
            messages=[
              {'role': 'system', 'content': instructions},
              {'role': 'system', 'content': "Context:\n" + closest_fact},
              #{'role': 'user', 'content': user_input},
              {'role': 'user', 'content': "Question: " + user_input + "\n Answer in one short sentence."},
            ],
            options={
                "temperature":0.0, # Low temperature for more focused responses
                "top_p": 0.5, # Narrow sampling for more predictable outputs,
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
