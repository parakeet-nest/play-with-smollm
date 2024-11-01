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

fact1 = '<RECORD>Philippe Charrière is a Solution Architect at Docker.</RECORD>'
fact2 = '<RECORD>Bob Morane is an expert with WASM.</RECORD>'
fact3 = '<RECORD>KeegOrg is a monster in the RPG Game "Chronicle of Swords".</RECORD>'
fact4 = '<RECORD>The best pizza in the world is the pineapple pizza.</RECORD>'

vec1 = ollama_client.embed(model='all-minilm', input=fact1)
vec2 = ollama_client.embed(model='all-minilm', input=fact2)
vec3 = ollama_client.embed(model='all-minilm', input=fact3)
vec4 = ollama_client.embed(model='all-minilm', input=fact4)

vectorDb = [
    (vec1['embeddings'][0], fact1),
    (vec2['embeddings'][0], fact2),
    (vec3['embeddings'][0], fact3),
    (vec4['embeddings'][0], fact4),

]

print("📚 Vector Database:", vectorDb)

while True:
    user_input = input("🤖 (type 'bye' to exit):> ")
    if user_input.lower() == "bye":
        print("👋 Goodbye!")
        break
    else:
        

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
              #{'role': 'user', 'content': user_input},.
              {'role': 'user', 'content': "Use ONLY the provided RECORD to answer the QUESTIOM. QUESTION:" + user_input},
            ],
            options={
                "temperature":0.0, # Low temperature for more focused responses
                "top_p": 0.5, # Narrow sampling for more predictable outputs,
                "stop": ["\n","</RECORD>"],  # Stop generation at these tokens
                #"stop": ["\n", "."],  # Stop generation at these tokens
            },
            stream=True,
        )

        for chunk in stream:
          print(chunk['message']['content'], end='', flush=True)

        print("\n")
