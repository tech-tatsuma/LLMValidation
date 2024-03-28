## Evaluatorとは？
LangSmithではevaluatorを指定することで、さまざまな評価指標を出力することができる。

evaluatorはそのほとんどがLLM自身による評価である。LLMに「このように評価して」とプロンプトで指示して、評価してもらう。

## Evaluators
LangSmithで選択できるevaluatorは次の通りである。
- Correctness
    - qa
    - context_qa
    - cot_qa
    - criteria

- conciseness
    - relevance
    - correctness
    - coherence
    - harmfulness
    - maliciousness
    - helpfulness
    - controversiality
    - misogyny
    - criminality
    - insensitivity
    - depth
    - creativity
    - detail
    - カスタム criteria
- distance
    - Embedding distance
    - String distance
- カスタム evaluator
## Correctness
Correctnessには、次のevaluatorがある
- qa
    
- context_qa
- cot_qa