from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith.evaluation import EvaluationResult, run_evaluator

# 外部のライブラリのインポート
from datasets.create_dataset import simple_create_dataset, create_runnalble, create_dataset_from_json

from dotenv import load_dotenv

load_dotenv()

# データセットの名前の設定
dataset_name = "kitei"
# データセットの作成
# client = simple_create_dataset(dataset_name)
client = create_dataset_from_json(r"C:\Users\DT5914\Desktop\all_datasets\dataset\kitei",dataset_name)
# チェーンの作成
chain = create_runnalble()

# カスタム評価関数を定義する。この関数は、予測結果に特定のフレーズが含まれているかを評価する
@run_evaluator
def must_mention(run, example) -> EvaluationResult:
    # 予測結果を取得
    prediction = run.outputs.get("output") or ""
    # 必須フレーズのリストを取得
    required = example.outputs.get("must_mention") or []
    # 全ての必須フレーズが予測結果に含まれているかどうかをチェック
    score = all(phrase in prediction for phrase in required)
    # 評価結果を返す
    return EvaluationResult(key="must_mention", score=score)

# 評価設定を定義。ここではカスタム評価関数と、既存の評価基準を使用
eval_config = RunEvalConfig(
    custom_evaluators=[
        must_mention,
        "embedding_distance", # cosine類似度
        ],
    evaluators=[
        # ヘルプフルネスに基づく既定の評価基準を使用
        "criteria",
        # 特定の評価基準(ここではconcisenessを使用)
        RunEvalConfig.Criteria("conciseness"),
        RunEvalConfig.Criteria("correctness"),
        RunEvalConfig.Criteria("coherence"),
        RunEvalConfig.Criteria("detail"),
        RunEvalConfig.Criteria("creativity"),
    ],
)
# データセット上で評価を実行。ここで、評価設定、チェーン、プロジェクト名、メタデータなどを設定
client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=chain,
    evaluation=eval_config,
    project_name="kitei",
    # Any experiment metadata can be specified here
    project_metadata={"version": "1.0.0"},
)