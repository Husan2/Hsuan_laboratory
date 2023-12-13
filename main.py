import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import qdrant_client

import requests
import json
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMMathChain

from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
serper_api_key = os.getenv("SERPAPI_API_KEY")
os.environ['QDRANT_COLLECTION_NAME'] = "my-collenction"

os.environ['QDRANT_URL'] = "..."
os.environ['QDRANT_API_KEY'] = "..."


# tool 搜尋
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "gl": "tw",
        "hl": "zh-tw",
        "num": 2

    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()

    # print("搜尋結果：", response_data)
    return response_data
# search("車力巨人有什麼能力")

# tool 算數
def math_calculator():
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

    return math_chain

# 向量庫
def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store
    
# 將聊天歷史紀錄加入向量庫前的準備(切割文字區塊)
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def main():

    vector_store = get_vector_store()

    # create vector chain 
    state_of_union = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
    
    tools = [
        Tool(
            name="Search",
            func=search, # 調用上面的搜尋函數
            description="這個工具最後才能使用，需要回答時事的問題，這個搜尋會很有幫助。只能繁體中文回答。"
        ),
        Tool(
            name="Calculator",
            func=math_calculator().run,
            description="當需要回答數學問題，這個工具會很有幫助。只能繁體中文回答。"
        ),
        Tool(
            name="Chat_history",
            func=state_of_union,
            description="回答 需要回憶歷史聊天內容 的問題時使用，只能繁體中文回答。"
        )

    ]


    # 定義系統訊息，用來傳遞給 Agent
    system_message = SystemMessage(
        content="""
        你是一位親切的專業助理。
        你只說繁體中文。
        一個問題思考不能跌代超過3次。
        你不能虛構信息，當遇到不知道問題的答案，就誠實說不知道。
        """
    )

    # Agent 代理參數 工具輸入模式
    agent_kwargs = {
        "extra_prompt_messages":[MessagesPlaceholder(variable_name="chat_history")],
        "system_message":system_message,
    }

    llm = ChatOpenAI(temperature=0, 
                     model="gpt-3.5-turbo", 
                     callbacks=[StreamingStdOutCallbackHandler()],
                     streaming= True,
                     verbose = False,
                     max_tokens=300)

    
    # 臨時記憶
    # ConversationSummaryBufferMemory 用於聊天歷史紀錄的所有逐字 word, 舊的內存會進行摘要(max_token_limit 數字限制是否為舊內容)
    memory = ConversationBufferMemory(memory_key="chat_history", 
                                             return_messages=True, 
                                             llm=llm, 
                                             max_token_limit=300,
                                             )
    

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose = False,
        agent_kwargs=agent_kwargs,
        memory=memory
    )
    
    while True:
        
        print("\n")
        user_input = input("使用者: ")
        print("\n機器人：")
        agent.run(input=user_input)

        # print(memory.load_memory_variables({}))
        output_text = str(memory.load_memory_variables({}))
        # print(type(output_text))

        with open('Lemon.txt', 'w', encoding='utf-8') as file:
            # 將print的內容寫入檔案
            file.write(output_text)

        with open("Lemon.txt", 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 內容先經過切割(超過500字時換下一個)
        texts = get_chunks(raw_text)
        
        # 將檔案內容上傳至 qdrant 雲端
        vector_store.add_texts(texts)

        # 當跟機器人說 '退出', '離開', '掰掰'時, 聊天紀錄才會上傳
        if user_input.lower() in ['退出', '離開', '掰掰']:
            # print("機器人: 再見!")
            break


if __name__ == '__main__':
    main()