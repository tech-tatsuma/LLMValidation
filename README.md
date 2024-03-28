# LangSmithを用いたLLM評価
## Overview
大規模言語モデル(LLM)は非決定論的な性質を持っているため、これらのモデルを用いたアプリケーションの適切な評価が非常に重要です。信頼性の高い評価フレームワークを持つことで、以下のような様々な利点があります。
- 様々なプロンプトを試して、性能向上のための変更が効果的かどうかを正確に判断できます。
- RAG（Retrieval-Augmented Generation）アーキテクチャの様々な部分を試行錯誤することが容易になります。
- 新しくリリースされたモデルにアップグレードする価値があるかどうかを素早く評価できます。
- 高コストのモデルの代わりに、より安価で高速なモデルを使用して同等の性能を達成できるかどうかを見極められます。
- APIが予期せずに変更された場合でも、その変更が背後にあるモデルにどのような影響を与えるかを追跡できます。
- アプリケーションが偏見を持ったり、不適切なレスポンスを生成しないことを信頼できます。
テストと評価は問題点を浮き彫りにし、異なるアーキテクチャの選択、改善されたモデルやプロンプトの使用、コードへの追加チェックの導入、またはその他の手段を通じて、問題に対処するための最適な方法を決定するのに役立ちます。

```bash
python eval.py
```
を実行すると
```bash
 Experiment Results:
        feedback.helpfulness  feedback.harmfulness  feedback.cliche feedback.must_mention error  execution_time                                run_id
count                   4.00                  4.00             4.00                     4     0            4.00                                     4
unique                   NaN                   NaN              NaN                     1     0             NaN                                     4
top                      NaN                   NaN              NaN                  True   NaN             NaN  35016ddc-5e9a-46d9-a1d6-130b089548bf
freq                     NaN                   NaN              NaN                     4   NaN             NaN                                     1
mean                    1.00                  0.00             0.75                   NaN   NaN            8.43                                   NaN
std                     0.00                  0.00             0.50                   NaN   NaN            1.70                                   NaN
min                     1.00                  0.00             0.00                   NaN   NaN            6.78                                   NaN
25%                     1.00                  0.00             0.75                   NaN   NaN            7.27                                   NaN
50%                     1.00                  0.00             1.00                   NaN   NaN            8.17                                   NaN
75%                     1.00                  0.00             1.00                   NaN   NaN            9.33                                   NaN
max                     1.00                  0.00             1.00                   NaN   NaN           10.61                                   NaN
```
のような結果が得られる