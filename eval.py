from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith.evaluation import EvaluationResult, run_evaluator

# 外部のライブラリのインポート
from datasets.create_dataset import simple_create_dataset, create_runnalble

from dotenv import load_dotenv

load_dotenv()

# データセットの名前の設定
dataset_name = ""
# データセットの作成
client = simple_create_dataset(dataset_name)
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
    custom_evaluators=[must_mention],
    evaluators=[
        # ヘルプフルネスに基づく既定の評価基準を使用
        "criteria",
        # 特定の評価基準(ここではharmfulnessを使用)
        RunEvalConfig.Criteria("harmfulness"),
        # カスタム評価基準を設定。
        RunEvalConfig.Criteria(
            {
                "cliche": "Are the lyrics cliche?"
                "Respond Y if they are, N if they're entirely unique."
            }
        ),
    ],
)
# データセット上で評価を実行。ここで、評価設定、チェーン、プロジェクト名、メタデータなどを設定
client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=chain,
    evaluation=eval_config,
    verbose=True,
    project_name="runnable-test-1",
    # Any experiment metadata can be specified here
    project_metadata={"version": "1.0.0"},
)