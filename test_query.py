import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say hello!"}]
)
print(response.choices[0].message.content)
