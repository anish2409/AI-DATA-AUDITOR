import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/llm_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = [
    {
        "case_id": 1,
        "prompt": "What should be done before training a model on a dataset with duplicate rows?",
        "reference_answer": "Duplicate rows should be investigated and removed or handled before model training.",
        "model_answer": "Duplicate rows should be removed or investigated before model training. Source: Data Quality Policy.",
        "reference_context": "Data Quality Policy: duplicate rows can distort training and should be removed or reviewed before modeling.",
        "expected_keywords": "duplicate,removed,training",
        "required_citation": "Data Quality Policy",
        "latency_ms": 950,
        "estimated_cost_usd": 0.002
    },
    {
        "case_id": 2,
        "prompt": "Can identifier columns like customer_id be used directly as model features?",
        "reference_answer": "Identifier columns should not be used directly because they can cause memorization and leakage risk.",
        "model_answer": "No. customer_id should be excluded because identifier columns can cause memorization and leakage. Source: Feature Safety Guide.",
        "reference_context": "Feature Safety Guide: identifier columns such as customer_id should be excluded from model features.",
        "expected_keywords": "identifier,excluded,leakage",
        "required_citation": "Feature Safety Guide",
        "latency_ms": 1100,
        "estimated_cost_usd": 0.003
    },
    {
        "case_id": 3,
        "prompt": "What does high PSI indicate in drift monitoring?",
        "reference_answer": "High PSI indicates significant distribution drift between baseline and current data.",
        "model_answer": "High PSI means the model is definitely accurate and ready for production.",
        "reference_context": "Drift Monitoring Guide: PSI above 0.25 indicates significant distribution drift.",
        "expected_keywords": "psi,drift,distribution",
        "required_citation": "Drift Monitoring Guide",
        "latency_ms": 890,
        "estimated_cost_usd": 0.002
    },
    {
        "case_id": 4,
        "prompt": "What should be checked if model recall is low?",
        "reference_answer": "Low recall means the model may be missing positive cases, so false negatives and class imbalance should be reviewed.",
        "model_answer": "Low recall means the model may miss positive cases. Review false negatives and class imbalance. Source: Model Evaluation Guide.",
        "reference_context": "Model Evaluation Guide: low recall indicates missed positive cases and should be reviewed with false negatives and imbalance.",
        "expected_keywords": "recall,false negatives,imbalance",
        "required_citation": "Model Evaluation Guide",
        "latency_ms": 760,
        "estimated_cost_usd": 0.002
    },
    {
        "case_id": 5,
        "prompt": "Is it safe to deploy a model when drift is significant?",
        "reference_answer": "No. Significant drift should be investigated before deployment or retraining decisions.",
        "model_answer": "Yes, significant drift is usually safe and does not need investigation.",
        "reference_context": "Deployment Policy: significant drift must be investigated before model deployment.",
        "expected_keywords": "drift,investigated,deployment",
        "required_citation": "Deployment Policy",
        "latency_ms": 1300,
        "estimated_cost_usd": 0.004
    },
    {
        "case_id": 6,
        "prompt": "What should be done with missing values before AI modeling?",
        "reference_answer": "Missing values should be analyzed and handled using appropriate imputation, removal, or business validation.",
        "model_answer": "Missing values should be handled before modeling through imputation, removal, or business validation. Source: Data Quality Policy.",
        "reference_context": "Data Quality Policy: missing values should be handled using imputation, removal, or validation before modeling.",
        "expected_keywords": "missing,imputation,validation",
        "required_citation": "Data Quality Policy",
        "latency_ms": 980,
        "estimated_cost_usd": 0.002
    },
    {
        "case_id": 7,
        "prompt": "What is the risk of using post-event status columns?",
        "reference_answer": "Post-event status columns may leak outcome information and inflate model performance unrealistically.",
        "model_answer": "Post-event status columns can leak outcome information and create unrealistic model performance. Source: Leakage Review Guide.",
        "reference_context": "Leakage Review Guide: post-event columns can leak target information and create unreliable model metrics.",
        "expected_keywords": "post-event,leak,outcome",
        "required_citation": "Leakage Review Guide",
        "latency_ms": 1020,
        "estimated_cost_usd": 0.003
    },
    {
        "case_id": 8,
        "prompt": "Should a dataset with high fairness disparity be automatically approved?",
        "reference_answer": "No. High fairness disparity should be reviewed before model approval.",
        "model_answer": "Yes. Fairness disparity does not affect AI model approval.",
        "reference_context": "Fairness Review Guide: high group disparity requires review before approval.",
        "expected_keywords": "fairness,disparity,review",
        "required_citation": "Fairness Review Guide",
        "latency_ms": 870,
        "estimated_cost_usd": 0.002
    }
]

df = pd.DataFrame(records)

output_path = OUTPUT_DIR / "llm_response_eval_dataset.csv"

df.to_csv(output_path, index=False)

print("LLM evaluation dataset created.")
print(f"Output path: {output_path}")
print(f"Rows: {len(df)}")