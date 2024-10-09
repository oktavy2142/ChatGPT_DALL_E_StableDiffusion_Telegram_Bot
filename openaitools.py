from os import getenv
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Tuple

load_dotenv()

client = AsyncOpenAI(
    api_key=getenv("OPENAI_API_KEY"),
)


class OpenAiTools:
    async def get_chatgpt(start: int, messages: List[Tuple[int, str, str, int]]):
        mess = []
        for i in range(start, len(messages)):
            mess.append({"role": messages[i][1], "content": messages[i][2]})

        try:
            response = await client.chat.completions.create(
                messages=mess,
                model="gpt-4o",
                max_tokens=16384,
                temperature=1,
            )

            if response and response.choices:
                return response.choices[0].message.content
            else:
                print("Invalid response format or empty response.")
                return None

        except Exception as e:
            print(f"Error while calling ChatGPT API: {str(e)}")
            return None

    async def get_dalle(prompt: str):
        try:
            response = await client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                n=1,
            )

            if response and response.data:
                return response.data[0].url
            else:
                print("Invalid response format or empty response.")
                return None

        except Exception as e:
            print(f"Error while calling DALL-E API: {str(e)}")
            return None
