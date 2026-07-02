# LLM Response Quality Evaluation Report

## Overall Summary

- Total cases evaluated: 8
- Average quality score: **66.61/100**
- Passed responses: 4
- Needs review: 0
- Failed responses: 4

## Failed / Risky Responses

- Case 2 | Score: 75.0 | Decision: Fail - Possible Contradiction
  - Risk: Possible contradiction with reference answer.
- Case 3 | Score: 34.85 | Decision: Fail
  - Risk: Required citation missing.
  - Risk: Low keyword coverage.
- Case 5 | Score: 17.12 | Decision: Fail - Possible Contradiction
  - Risk: Possible contradiction with reference answer.
  - Risk: Required citation missing.
  - Risk: Low keyword coverage.
- Case 8 | Score: 35.33 | Decision: Fail - Possible Contradiction
  - Risk: Possible contradiction with reference answer.
  - Risk: Required citation missing.
  - Risk: Low keyword coverage.

## Full Evaluation Table

|   case_id |   keyword_coverage | matched_keywords                   | citation_present   |   reference_overlap | contradiction_flag   |   latency_ms | latency_risk   |   estimated_cost_usd | cost_risk   |   quality_score | final_decision                |
|----------:|-------------------:|:-----------------------------------|:-------------------|--------------------:|:---------------------|-------------:|:---------------|---------------------:|:------------|----------------:|:------------------------------|
|         1 |             1      | duplicate, removed, training       | True               |              0.8333 | False                |          950 | Low            |                0.002 | Low         |           95    | Pass                          |
|         2 |             1      | identifier, excluded, leakage      | True               |              0.6667 | True                 |         1100 | Medium         |                0.003 | Medium      |           75    | Fail - Possible Contradiction |
|         3 |             0.3333 | psi                                | False              |              0.2727 | False                |          890 | Low            |                0.002 | Low         |           34.85 | Fail                          |
|         4 |             1      | recall, false negatives, imbalance | True               |              0.7222 | False                |          760 | Low            |                0.002 | Low         |           91.67 | Pass                          |
|         5 |             0.3333 | drift                              | False              |              0.1818 | True                 |         1300 | Medium         |                0.004 | Medium      |           17.12 | Fail - Possible Contradiction |
|         6 |             1      | missing, imputation, validation    | True               |              0.7143 | False                |          980 | Low            |                0.002 | Low         |           91.43 | Pass                          |
|         7 |             1      | post-event, leak, outcome          | True               |              0.75   | False                |         1020 | Medium         |                0.003 | Medium      |           92.5  | Pass                          |
|         8 |             0.6667 | fairness, disparity                | False              |              0.4    | True                 |          870 | Low            |                0.002 | Low         |           35.33 | Fail - Possible Contradiction |

## Interpretation

- High quality score means the response is closer to the reference answer.
- Missing citation reduces trust even if the answer is partly correct.
- Contradiction risk is treated as a serious failure.
- This module is a rule-based evaluator, not a final human judgment.
- In production, this can be combined with human review or another evaluator model.