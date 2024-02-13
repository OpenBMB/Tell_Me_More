from openai import OpenAI
import json5
import time

client = OpenAI(
    api_key="<api-key>",
    base_url="<base-url>",
)

def get_embedding(text:str, model="text-embedding-ada-002"):
   text = text.replace("\n", " ").strip()
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def gpt_chatcompletion(messages):
    rounds = 0
    while True:
        rounds += 1
        try:
            print("Chat Completion ...")
            response = client.chat.completions.create(
                model="gpt-4-1106",
                messages=messages,
                temperature=0.7,
                n=1,
            )
            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            print(f"Chat Generation Error: {e}")
            time.sleep(5)
            if rounds > 3:
                raise Exception("Chat Completion failed too many times")
