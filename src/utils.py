from openai import OpenAI
import json5
import time

client = OpenAI(
    api_key="sk-Lb7yCt7UOCAMQenfF9Ea709c09524eDaA2Bb099dDd523e63",
    base_url="https://sailaoda.cn/v1",
)

def get_embedding(text:str, model="text-embedding-ada-002"):
   text = text.replace("\n", " ").strip()
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def gpt_toolcall(messages, function, function_name):
    rounds = 0
    while True:
        rounds += 1
        try:
            print("Chat Tool Call ...")
            response = client.chat.completions.create(
                model="gpt-4-1106",
                messages=messages,
                tools=[{"type": "function", "function": function}],
                tool_choice={"type": "function", "function": {"name": function_name}},
                temperature=0.7,
                n=1,
            )
            function_args = json5.loads(response.choices[0].message.tool_calls[0].function.arguments)
            content = response.choices[0].message.content
            return function_args, content
        except Exception as e:
            print(f"Tool Call Error: {e}")
            time.sleep(5)
            if rounds > 3:
                raise Exception("Chat Tool Call failed too many times")

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
