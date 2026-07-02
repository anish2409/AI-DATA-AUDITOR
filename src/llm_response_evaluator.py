import pandas as pd
import re
from pathlib import Path


INPUT_PATH = "data/llm_eval/llm_response_eval_dataset.csv"

OUTPUT_SCORE_PATH = "data/exports/llm_response_quality_scores.csv"

OUTPUT_REPORT_PATH = "reports/llm_response_quality_report.md"


NEGATIVE_PATTERNS = [
    "not",
    "no",
    "should not",
    "cannot",
    "must not",
    "do not"
]


POSITIVE_PATTERNS = [
    "yes",
    "safe",
    "approved",
    "ready",
    "definitely",
    "does not need"
]


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9\s\-]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_keyword_coverage(answer, expected_keywords):
    answer_text = normalize_text(answer)

    keywords = [
        keyword.strip().lower()
        for keyword in str(expected_keywords).split(",")
        if keyword.strip()
    ]

    if not keywords:
        return 0, []

    matched_keywords = []

    for keyword in keywords:
        if keyword in answer_text:
            matched_keywords.append(keyword)

    coverage = len(matched_keywords) / len(keywords)

    return round(coverage, 4), matched_keywords


def check_required_citation(answer, required_citation):
    answer_text = normalize_text(answer)

    citation_text = normalize_text(required_citation)

    if citation_text and citation_text in answer_text:
        return True

    return False


def calculate_reference_overlap(model_answer, reference_answer):
    model_tokens = set(normalize_text(model_answer).split())

    reference_tokens = set(normalize_text(reference_answer).split())

    if not reference_tokens:
        return 0

    overlap = model_tokens.intersection(reference_tokens)

    return round(len(overlap) / len(reference_tokens), 4)


def detect_contradiction_risk(reference_answer, model_answer):
    reference_text = normalize_text(reference_answer)

    model_text = normalize_text(model_answer)

    reference_has_negative = any(
        pattern in reference_text for pattern in NEGATIVE_PATTERNS
    )

    model_has_positive = any(
        pattern in model_text for pattern in POSITIVE_PATTERNS
    )

    reference_has_positive = any(
        pattern in reference_text for pattern in POSITIVE_PATTERNS
    )

    model_has_negative = any(
        pattern in model_text for pattern in NEGATIVE_PATTERNS
    )

    if reference_has_negative and model_has_positive:
        return True

    if reference_has_positive and model_has_negative:
        return True

    return False


def latency_risk(latency_ms):
    if latency_ms <= 1000:
        return "Low"

    if latency_ms <= 1500:
        return "Medium"

    return "High"


def cost_risk(cost_usd):
    if cost_usd <= 0.002:
        return "Low"

    if cost_usd <= 0.004:
        return "Medium"

    return "High"


def final_decision(score, contradiction_flag):
    if contradiction_flag:
        return "Fail - Possible Contradiction"

    if score >= 85:
        return "Pass"

    if score >= 65:
        return "Needs Review"

    return "Fail"


def evaluate_llm_responses(df):
    rows = []

    for _, row in df.iterrows():
        keyword_coverage, matched_keywords = calculate_keyword_coverage(
            answer=row["model_answer"],
            expected_keywords=row["expected_keywords"]
        )

        citation_present = check_required_citation(
            answer=row["model_answer"],
            required_citation=row["required_citation"]
        )

        reference_overlap = calculate_reference_overlap(
            model_answer=row["model_answer"],
            reference_answer=row["reference_answer"]
        )

        contradiction_flag = detect_contradiction_risk(
            reference_answer=row["reference_answer"],
            model_answer=row["model_answer"]
        )

        latency_label = latency_risk(row["latency_ms"])

        cost_label = cost_risk(row["estimated_cost_usd"])

        score = 0

        score += keyword_coverage * 35

        score += reference_overlap * 30

        if citation_present:
            score += 20

        if not contradiction_flag:
            score += 15

        quality_score = round(score, 2)

        rows.append({
            "case_id": row["case_id"],
            "keyword_coverage": keyword_coverage,
            "matched_keywords": ", ".join(matched_keywords),
            "citation_present": citation_present,
            "reference_overlap": reference_overlap,
            "contradiction_flag": contradiction_flag,
            "latency_ms": row["latency_ms"],
            "latency_risk": latency_label,
            "estimated_cost_usd": row["estimated_cost_usd"],
            "cost_risk": cost_label,
            "quality_score": quality_score,
            "final_decision": final_decision(
                score=quality_score,
                contradiction_flag=contradiction_flag
            )
        })

    return pd.DataFrame(rows)


def generate_report(results_df, output_path):
    output = []

    output.append("# LLM Response Quality Evaluation Report")
    output.append("")

    total_cases = len(results_df)

    pass_count = len(results_df[results_df["final_decision"] == "Pass"])

    review_count = len(results_df[results_df["final_decision"] == "Needs Review"])

    fail_count = len(
        results_df[
            results_df["final_decision"].str.contains("Fail", na=False)
        ]
    )

    avg_score = round(results_df["quality_score"].mean(), 2)

    output.append("## Overall Summary")
    output.append("")
    output.append(f"- Total cases evaluated: {total_cases}")
    output.append(f"- Average quality score: **{avg_score}/100**")
    output.append(f"- Passed responses: {pass_count}")
    output.append(f"- Needs review: {review_count}")
    output.append(f"- Failed responses: {fail_count}")

    output.append("")
    output.append("## Failed / Risky Responses")
    output.append("")

    risky = results_df[
        results_df["final_decision"].str.contains("Fail|Review", na=False)
    ]

    if risky.empty:
        output.append("- No risky responses detected.")
    else:
        for _, row in risky.iterrows():
            output.append(
                f"- Case {row['case_id']} | Score: {row['quality_score']} | Decision: {row['final_decision']}"
            )

            if row["contradiction_flag"]:
                output.append("  - Risk: Possible contradiction with reference answer.")

            if not row["citation_present"]:
                output.append("  - Risk: Required citation missing.")

            if row["keyword_coverage"] < 0.7:
                output.append("  - Risk: Low keyword coverage.")

    output.append("")
    output.append("## Full Evaluation Table")
    output.append("")
    output.append(results_df.to_markdown(index=False))

    output.append("")
    output.append("## Interpretation")
    output.append("")
    output.append("- High quality score means the response is closer to the reference answer.")
    output.append("- Missing citation reduces trust even if the answer is partly correct.")
    output.append("- Contradiction risk is treated as a serious failure.")
    output.append("- This module is a rule-based evaluator, not a final human judgment.")
    output.append("- In production, this can be combined with human review or another evaluator model.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    Path(output_path).write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    results_df = evaluate_llm_responses(df)

    Path("data/exports").mkdir(parents=True, exist_ok=True)

    Path("reports").mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        OUTPUT_SCORE_PATH,
        index=False
    )

    generate_report(
        results_df=results_df,
        output_path=OUTPUT_REPORT_PATH
    )

    avg_score = round(results_df["quality_score"].mean(), 2)

    fail_count = len(
        results_df[
            results_df["final_decision"].str.contains("Fail", na=False)
        ]
    )

    print("LLM response quality evaluation completed.")
    print(f"Average quality score: {avg_score}/100")
    print(f"Failed responses: {fail_count}")
    print(f"Report saved: {OUTPUT_REPORT_PATH}")
    print(f"Scores saved: {OUTPUT_SCORE_PATH}")