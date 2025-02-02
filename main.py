import chainlit as cl
# from langchain_ollama.chat_models import ChatOllama
from langchain_community.llms.llamafile import Llamafile 
from langchain_core.prompts import PromptTemplate

# Initialize the deepseek model
llm = Llamafile()

# Define a prompt template for the LLM chain
prompt = PromptTemplate(
    input_variables=["input"],
    template="{input}"
)

# Create an LLMChain with the deepseek model and the defined prompt template
llm_chain = prompt | llm

@cl.on_message
async def main(message: str):
    # Process the input message using the llm chain
    response = await cl.make_async(llm_chain.invoke)({"input": message})
    # Send the response back to the user
    await cl.Message(content=response).send()
