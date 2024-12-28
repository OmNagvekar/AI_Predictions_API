from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from gradio import ChatInterface

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-base-en',cache_folder='../embed_model;',device='cuda')
LLm = Ollama(model='phi3:mini')
data = SimpleDirectoryReader(input_dir='./data').load_data()



class Swachta_Chatbot:
    
    def __init__(self):
        self._index = VectorStoreIndex.from_documents(data,embed_model=Embed_model)
        self.query_engine = self._index.as_query_engine(llm=LLm)
    def get_response(self,message):
        return self.query_engine.query(message)




if __name__ == '__main__':
    chat_bot = Swachta_Chatbot()


    def chat(message:str,history):
        response = chat_bot.get_response(message).response
        return response


    demo = ChatInterface(fn=chat, type="messages", examples=["hello", "hola", "merhaba"], title="Cyber Bot")
    demo.launch(share=True)
