from langsmith import Client
from langchain import chat_models, prompts
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.retriever import BaseRetriever
from langchain.docstore.document import Document
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langchain.evaluation import load_evaluator
from langchain.smith import RunEvalConfig

import uuid
import os
import json

uid = uuid.uuid4()

# ディレクトリ内の全てのjsonファイルを読み込む
json_files_directory = '/data'  # ディレクトリパス
source_files_directory = '/source'  # ディレクトリパス
examples = []

# ディレクトリ内の全JSONファイルを探索
for filename in os.listdir(json_files_directory):
    if filename.endswith('.json'):
        # JSONファイルのフルパス
        json_file_path = os.path.join(json_files_directory, filename)
        
        # JSONファイルを開いて内容を読み込む
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # 必要な情報を抽出
            question = data.get('question', '')
            answer = data.get('answer', '')
            source_filename = data.get('source', '')
            
            # sourceファイルのパスを生成
            source_file_path = os.path.join(source_files_directory, source_filename)
            
            # sourceファイルが存在する場合、その内容を読み取る
            page_content = ''
            if os.path.exists(source_file_path):
                with open(source_file_path, 'r', encoding='utf-8') as source_file:
                    page_content = source_file.read()
                    
            # examplesリストにデータを追加
            examples.append({
                "inputs": {
                    "question": question,
                    "documents": [
                        {
                            "metadata": {},  # metadataは空とする
                            "page_content": page_content,
                        }
                    ],
                },
                "outputs": {
                    "label": answer,
                },
            })

# langsmithクライアントの初期化
client = Client()
dataset_name = f"Faithfulness Example - {uid}"

# データセットを作成
dataset = client.create_dataset(dataset_name=dataset_name)
client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id,
)

# 忠実度評価のためのカスタムリトリーバーを定義
class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="Example")]


# 評価するためのchainを作成
response_synthesizer = prompts.ChatPromptTemplate.from_messages(
    [
        ("system", """以下の社内規定文書に基づいて規程に関する質問と回答を生成してください。
                    質問は文書の内容に基づいて生成してください。但し、一般社員が質問することを考慮して、具体的な質問を生成してください。
                    質問も回答も日本語で生成してください。

                    社内規定文書:\n{documents}"""),
        ("user", "{question}"),
    ]
) | chat_models.ChatAnthropic(model="claude-2", max_tokens=1000)

chain = {
    "documents": MyRetriever(),
    "qusetion": RunnablePassthrough(),
} | response_synthesizer

# 忠実度評価のためのカスタム評価関数を定義
class FaithfulnessEvaluator(RunEvaluator):
    def __init__(self):
        self.evaluator = load_evaluator(
            "labeled_score_string",
            criteria={
                "faithful": "How faithful is the submission to the reference context?"
            },
            normalize_by=10,
        )

    def evaluate_run(self, run, example) -> EvaluationResult:
        res = self.evaluator.evaluate_strings(
            prediction=next(iter(run.outputs.values())),
            input=run.inputs["question"],
            # 文書を参照コンテキストとして扱う
            reference=example.inputs["documents"],
        )
        return EvaluationResult(key="labeled_criteria:faithful", **res)
    
# 評価設定
eval_config = RunEvalConfig(
    evaluators=["qa"],
    custom_evaluators=[FaithfulnessEvaluator()],
    input_key="question",
)

# データセット上での実行と評価
results = client.run_on_dataset(
    llm_or_chain_factory=response_synthesizer,
    dataset_name=dataset_name,
    evaluation=eval_config,
)