from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

from dotenv import load_dotenv

load_dotenv()

# 外部のライブラリのインポート
from datasets.create_dataset import create_runnalble

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
    project_name="kitei",
)