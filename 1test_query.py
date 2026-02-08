#!/usr/bin/env python3

from openai import OpenAI

# Создаем клиент для подключения к vLLM серверу
client = OpenAI(
    base_url="http://localhost:8000/v1",  # URL вашего vLLM сервера
    api_key="EMPTY"  # vLLM не требует настоящий API ключ
)

# Отправляем запрос
response = client.chat.completions.create(
    model="deepseek-ai/deepseek-coder-1.3b-instruct",
    messages=[
        {"role": "system", "content": "You are helpfull c++ assistant "},
        {"role": "user", "content": "Can you explain me what some c++ code doing?"}
    ],
    temperature=0.3,
    max_tokens=500
)

# Выводим ответ
print(response.choices[0].message.content)