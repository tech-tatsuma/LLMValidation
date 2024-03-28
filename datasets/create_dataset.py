from langsmith import Client

# langchain関連のライブラリのインポート
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import pandas as pd

# データセットを作成する関数(スクリプトに直接入力)
def simple_create_dataset(dataset_name: str):
    # 入力はモデルに提供されるため、何を生成するかを知ることができる。
    dataset_inputs = [
    ("What is the largest mammal?", "The blue whale"),
    ("What do mammals and birds have in common?", "They are both warm-blooded"),
    ("What are reptiles known for?", "Having scales"),
    ("What's the main characteristic of amphibians?", "They live both in water and on land"),
    ]

    client = Client() # LangSmithのクライアントのインスタンスを作成

    # 入力をデータセットに保存することで、共有される一連の例に対してチェーンとLLMを実行できる
    dataset = client.create_dataset(
        dataset_name=dataset_name, # データセット名を指定
        description="ここにデータセットの説明", # データセットの説明
    )
    for input_prompt, output_answer in dataset_inputs:
        client.create_example(
        inputs={"question": input_prompt},
        outputs={"answer": output_answer},
        dataset_id=dataset.id,
    )

    # clientを返す
    return client

# runnableの作成
def create_runnalble():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([("human", "Spit some bars about {question}.")])
    chain = prompt | llm | StrOutputParser()

    return chain