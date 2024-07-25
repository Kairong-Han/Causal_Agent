
import random
import threading

import openai
import os

openai.api_base = ""
class Keypool():
    def __init__(self):
        self.api_keys = [
            '',
        ]
        self.counter = 0
        self.lock = threading.Lock()

    def getkey(self):
        self.lock.acquire()
        key_ret = self.api_keys[self.counter]
        self.counter = (self.counter + 1) % len(self.api_keys)
        self.lock.release()
        return key_ret


class ChatGPT:
    def __init__(self, model="gpt-4", key="", conversation_list=[],keypool = None):
        self.model = model
        self.conversation_list = conversation_list
        self.lock = threading.Lock()
        openai.api_key = ""
        self.client = openai.OpenAI(
            base_url="",
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key="",
        )

    def call(self, prompt,model = "gpt-4"):
        answer = None
        self.conversation_list.append(
            {"role": "system", "content": "You are a helpful instruction-following assistant."})
        self.conversation_list.append({"role": "user", "content": prompt})
        try:
            answer = ""

            response = self.client.chat.completions.create(model=model, messages=self.conversation_list, temperature=0.5)
            answer = response.choices[0].message.content.strip()
            self.conversation_list = []
        except Exception as e:
            print("Call Openai API Error:", e)
            raise e
        return answer

gpt = ChatGPT()