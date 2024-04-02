from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import uuid

load_dotenv()

uid = uuid.uuid4()
# runnableの作成
def create_runnalble():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_messages([("human", "Spit some bars about {question}.")])
    chain = prompt | llm | StrOutputParser()

    return chain

# チェーンの作成
chain = create_runnalble()

evaluation_config = RunEvalConfig(
    # ここで指定
    evaluators=[
        # Correctness
        "qa",
        # "context_qa", # コンテキストを考慮した指標
        # "cot_qa",
        # criteria
        RunEvalConfig.Criteria("conciseness"),
        RunEvalConfig.Criteria("coherence"),
        RunEvalConfig.Criteria("helpfulness"),
        RunEvalConfig.Criteria("insensitivity"),
        RunEvalConfig.Criteria("creativity"),
        RunEvalConfig.Criteria("detail"),
        # distance
        "embedding_distance", # cosine類似度
    ]
)

client = Client()

run_on_dataset(
    dataset_name="kitei",
    llm_or_chain_factory=chain,
    client=client,
    evaluation=evaluation_config,
    project_name=f"kitei - {uid}",
)