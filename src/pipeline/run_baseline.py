import csv
import sys

from src.pipeline.pipeline import ask

QUESTIONS = [
    {
        "question": "What is this document about?",
        "ground_truth": "An agreement on organizing internships/practical training for students and cadets.",
    },
    {
        "question": "What type of education program is mentioned in the document?",
        "ground_truth": "Secondary specialized (vocational) education programs.",
    },
    {
        "question": "Which institution is named in the document?",
        "ground_truth": "Polotsk State Economic College.",
    },
    {
        "question": "What kind of document is this, in legal terms?",
        "ground_truth": "A contract / agreement (договор).",
    },
    {
        "question": "Who are the parties to the agreement?",
        "ground_truth": "The educational institution (college) and an organization/enterprise hosting the internship.",
    },
    {
        "question": "What is the educational institution required to do according to the agreement?",
        "ground_truth": "Familiarize itself with the internship curriculum, ensure it is carried out, and provide access to legal acts, technical regulations, and other documentation available at the organization.",
    },
    {
        "question": "What must be prepared at the end of the internship?",
        "ground_truth": "Reporting documentation, including a diary and a written report, plus a performance characteristic/evaluation for each student.",
    },
    {
        "question": "What does the host organization need to monitor during the internship?",
        "ground_truth": "Compliance with the internship program and related rules.",
    },
    {
        "question": "What section of the document contains the parties' contact details?",
        "ground_truth": "The 'Addresses, details, and signatures of the parties' section.",
    },
    {
        "question": "Does the document specify additional terms beyond the main obligations?",
        "ground_truth": "Yes, it has separate sections for 'other conditions' and 'additional conditions'.",
    },
    {
        "question": "What kind of students does this agreement cover?",
        "ground_truth": "Students and cadets enrolled in secondary specialized (vocational) education programs.",
    },
    {
        "question": "What is the capital of France?",
        "ground_truth": "NOT ANSWERABLE FROM THIS DOCUMENT — used to check that the system says so instead of guessing.",
    },
]

OUTPUT_PATH = "data/baseline.csv"


def run_baseline():
    rows = []

    for item in QUESTIONS:
        question = item["question"]
        ground_truth = item["ground_truth"]

        result = ask(question)
        sources_str = "; ".join(
            f"{s['source']} p.{s['page_num']}" for s in result["sources"]
        )

        rows.append({
            "question": question,
            "answer": result["answer"],
            "ground_truth": ground_truth,
            "sources": sources_str,
        })

        print(f"Q: {question}")
        print(f"A: {result['answer'][:150]}...")
        print()

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "ground_truth", "sources"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_baseline()