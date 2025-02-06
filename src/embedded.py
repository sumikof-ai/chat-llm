from dotenv import load_dotenv
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()
def openai_embeddings():
    """
    [[0.009695346467196941, 0.004457102622836828, -0.02351302094757557, -0.011060703545808792, -0.010876905173063278,
      0.01851108856499195, -0.051804788410663605, -0.01528149377554655, 0.02256777323782444,
    ...
    """
    embedder = OpenAIEmbeddings()
    fuji = embedder.embed_documents(["富士山は高い"])
    print(fuji)


def llamafile_openai_embeddings():
    """
    .\llava-v1.5-7b-q4.llamafile.exe -ngl 9999 -m .\Llama-3-Swallow-8B-Instruct-v0.1.Q4_K_M.iMatrix.gguf
    
    [[0.00026159945991821587, 0.012206138111650944, 0.00540942745283246, -0.007807696703821421, -0.004004584159702063, 
    0.008286013267934322, 0.004262338858097792, 0.007552321534603834, -0.0021030197385698557, -0.017583873122930527,
    ...
    """
    embedder = OpenAIEmbeddings(base_url="http://localhost:8080/v1")
    fuji = embedder.embed_documents(["富士山は高い"])
    print(fuji)

if __name__ == '__main__':
    openai_embeddings()