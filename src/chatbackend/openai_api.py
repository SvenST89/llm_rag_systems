import os
import getpass
import tiktoken_ext.openai_public
import inspect
import hashlib
import urllib.request
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import dotenv
from dotenv import load_dotenv
import logging


def openai_chat_llm(deployment_name="gpt-4o-mini", maxtokens=3500, temperature=0):
    found_dotenv = dotenv.find_dotenv()
    if found_dotenv:
        load_dotenv()
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    else:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    
    llm = ChatOpenAI(
        model=deployment_name,
        temperature=temperature,
        max_tokens=maxtokens,
        max_retries=2,
        api_key=OPENAI_API_KEY
    )
    
    return llm
        
def openai_embedder(deployment_name="text-embedding-3-small"):
    """In order to use the OpenAI embedding models locally, too, we need to download the required tiktoken-file and store it in a tiktoken cache dir manually.
    The usual encoder we want is cl100k_base. We can download this encoder from an Azure Blob-url 'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken'.
    Check the source here: https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer
    Args
    ----
        deployment_name (str, optional): which embedder model from openai 'https://openai.com/api/pricing/'. Defaults to "text-embedding-3-small".
    
    Help
    ----
        Call the embedder:
            embedder = openai_embedder(deployment_name="...")
        Embed documents:
            vecs = embedder.embed_documents(docs)
        Embed query:
            embeddings_q = embedder.embed_query("Hallo, Welt!")
    """
    found_dotenv = dotenv.find_dotenv()
    if found_dotenv:
        load_dotenv()
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    else:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    
    #--------------------------------------------------------#
    # Find the correct Blob-Url from Azure to download the
    # encoder required for embeddings!
    
    # The encoder we want is cl100k_base, we see this as a possible function in
    #print(dir(tiktoken_ext.openai_public))
    
    # The url should be in the 'load_tiktoken_bpe' function call
    #print(inspect.getsource(tiktoken_ext.openai_public.cl100k_base))
    #--------------------------------------------------------#
    blobpath = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
    
    tiktoken_cache_dir = "/workspaces/20241123_RAG_System/tiktoken_cache_dir"
    logging.info(f"tiktoken_cache_dir: {tiktoken_cache_dir}")
    if not os.path.exists(tiktoken_cache_dir):
        os.makedirs(tiktoken_cache_dir, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
    
    try:
        # download the right tiktoken encoder
        urllib.request.urlretrieve(blobpath, f"{tiktoken_cache_dir}/{cache_key}")
    except:
        logging.error(f"""There seems to have gone something wrong by downloading the tiktoken encoder model cl100k_base for using OpenAIEmbeddings()! \ 
                      Please download this encoder model from the blobpath-url {blobpath} and store it under {tiktoken_cache_dir}. \ 
                      Afterwards, rename this encoder file to a hashed name with 'hashlib.sha1(blobpath.encode()).hexdigest()'!"""
            )
    
    embedder = OpenAIEmbeddings(
        model=deployment_name,
        api_key=OPENAI_API_KEY
    )
    
    return embedder

# query = "Was ist der aktuelle Zinssatz der EZB?"
# embedder = openai_embedder()
# qvec = embedder.embed_query(query)
# print(qvec)