"""Generate eval_set.jsonl: a 30-item evaluation set for the IRS RAG assistant.

Each item is a real question answerable from data/corpus.jsonl, paired with the
id of the gold passage that holds the answer and a short reference answer. The set
is representative on purpose: common questions, year-sensitive ones (2024 vs 2025),
and two unanswerable ones (gold_id null) so we can check the assistant correctly
says it cannot find the answer.

The eval set is an ASSET: rerun the retriever, model, or a fine-tune later and
score against this same set to see whether a change helped.

Run:  python make_eval_set.py   ->  writes eval_set.jsonl next to this file
"""

import json
import pathlib

# (question, gold_id or None, reference answer). gold_id must be an id in
# corpus.jsonl; None marks an unanswerable question.
ITEMS = [
    ("What is the standard deduction for a single filer in 2025?", "p501-2025-stdded", "15,000."),
    ("What is the 2025 standard deduction for married filing jointly?", "p501-2025-stdded", "30,000."),
    ("What was the standard deduction for a single filer in 2024?", "p501-2024-stdded", "14,600."),
    ("The standard deduction went up this year; what is it now for a single filer?", "p501-2025-stdded", "15,000 for 2025."),
    ("How many filing statuses are there?", "p501-2025-status", "Five."),
    ("Which filing status categories exist?", "p501-2025-status", "Single, married filing jointly, married filing separately, head of household, and qualifying surviving spouse."),
    ("Does everyone have to file a federal income tax return?", "p501-2025-mustfile", "No; it depends on income, filing status, and age."),
    ("What is a dependent?", "p501-2025-dependent", "A qualifying child or a qualifying relative."),
    ("What fraction of AGI must medical expenses exceed before they are deductible?", "p502-2025-threshold", "7.5 percent of AGI."),
    ("Where do you claim medical expense deductions?", "p502-2025-threshold", "On Schedule A, as an itemized deduction."),
    ("If my AGI is 100,000, above what amount are medical expenses deductible?", "p502-2025-threshold", "Above 7,500, which is 7.5 percent of AGI."),
    ("Can I include dental treatment in medical expenses?", "p502-2025-dental", "Yes, dental treatment counts as a medical expense."),
    ("Are fees paid to doctors deductible medical expenses?", "p502-2025-deductible", "Yes, fees to doctors are deductible medical expenses."),
    ("Can I deduct nonprescription drugs?", "p502-2025-notdeductible", "Generally no."),
    ("What is the 2025 HSA contribution limit for self-only coverage?", "p969-2025-limit", "4,300."),
    ("What is the 2025 HSA contribution limit for family coverage?", "p969-2025-limit", "8,550."),
    ("What was the 2024 HSA self-only contribution limit?", "p969-2024-limit", "4,150."),
    ("Who is eligible to contribute to an HSA?", "p969-2025-eligible", "Someone covered by a high-deductible health plan."),
    ("Do I need a special kind of health plan to open an HSA?", "p969-2025-eligible", "Yes, a high-deductible health plan."),
    ("Are HSA distributions for qualified medical expenses taxed?", "p969-2025-qualified", "No, qualified medical distributions are tax-free."),
    ("Can I deduct charitable contributions?", "p526-2025-deduct", "Yes, gifts to qualified organizations."),
    ("Is there a limit on how much I can deduct for charitable gifts?", "p526-2025-limit", "Yes, capped at a percentage of AGI."),
    ("Can I deduct unlimited charitable donations?", "p526-2025-limit", "No, the deduction is capped at a share of AGI."),
    ("What record do I need for a cash donation?", "p526-2025-records", "A bank record or a written acknowledgment."),
    ("What is the debt limit for deducting home mortgage interest?", "p936-2025-limit", "750,000 of home acquisition debt for loans after 2017."),
    ("What are points on a home mortgage?", "p936-2025-points", "Charges paid to obtain a mortgage; a form of prepaid interest."),
    ("Are mortgage points a form of interest?", "p936-2025-points", "Yes, they are prepaid interest."),
    ("What qualifies as home mortgage interest?", "p936-2025-deductible", "Interest on a loan secured by your home."),
    # Unanswerable: the corpus has no such content, so the assistant should decline.
    ("What is the federal corporate income tax rate?", None, "Not found in the documents."),
    ("How do I form an S corporation?", None, "Not found in the documents."),
]


def main():
    here = pathlib.Path(__file__).parent
    with (here / "eval_set.jsonl").open("w") as f:
        for i, (q, gold, ref) in enumerate(ITEMS, 1):
            f.write(json.dumps({"id": f"q{i:02d}", "question": q,
                                "gold_id": gold, "reference": ref}) + "\n")
    answerable = sum(1 for _, g, _ in ITEMS if g)
    print(f"wrote {len(ITEMS)} items ({answerable} answerable, "
          f"{len(ITEMS) - answerable} unanswerable) to eval_set.jsonl")


if __name__ == "__main__":
    main()
