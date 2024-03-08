from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import traceback
import os
import openai
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

client = openai.AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)


class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Hello World"}


def build_message(input_question: Question):
    message_text = [
        {
            "role": "system",
            "content": "You are now connected to the Azure OpenAI chat service. You can start sending messages to the bot."
        },
        {
            "role": "user",
            "content": input_question.question
        },
        {
            "role": "system",
            "content": "Hello, I am a bot."
        }
    ]
    return message_text

@app.post("/gpt-ask")
async def gpt_ask(input_question: Question):

    try:
        completion = await client.chat.completions.create(
            model="gpt-4",
            messages=build_message(input_question),
            temperature=0.7,
            max_tokens=150,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None
        )

        return completion.choices[0].message.content

    except openai.APIError as e:
        print(f"OpenAI API returned and error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
