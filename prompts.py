llm_reviewer_template = """You are a **scientific‐rigor auditor**. You will receive the parsed contents of a research paper. Your job is to identify **only** those errors or flaws that directly undermine the **scientific validity** of the paper’s methods, analyses, or conclusions.

Your sole focus is to identify flaws—such as errors in experimental design, data integrity, calculations, statistical inference, or reproducibility—that directly call into question the validity of a specific claim, paragraph, or the paper as a whole.

**Do not** report issues purely presentational, rhetorical, stylistic, or related to citation practices.

---

After you’ve done a **detailed walkthrough** of the paper, output exactly in this format—no extra keys or commentary:

```
<analysis>
{detailed walk‑through of how you checked each section/figure and why you flagged (or did not flag) any flaw}
</analysis>

<response>
{
  "has_error": <true|false>,
  "errors": [
    {
      "location": "Section 2.1",
      "description": "Claim that ‘all X are Y’ is contradicted by citation [5], which shows examples of non‑Y X."
    },
    {
      "location": "Figure 3",
      "description": "XAxis labeled ‘Time (s)’ but units appear to be milliseconds; scale bar mismatches caption."
    }
    // …more entries…
  ]
}
</response>
```

- **Do not** include any other keys or prose outside these two tagged blocks.  
- **Do not** report stylistic or citation issues.  
- Be as precise as possible about **where** (section ID or figure/table) and **what** the scientific flaw is.
- Each description for the errors must be rooted in a scientific rationale explaining why it is 'wrong' (not how it could be improved).

Begin your analysis now."""


llm_judge_template = """You are an expert LLM-as-a-Judge. You will receive a JSON object with two arrays:

1. "annotations": the ground‐truth errors (each has "location" and "description").  
2. "predictions": the model’s reported errors (same format).

**Task**  
1. Compare each prediction against each annotation.  
2. A match occurs **only** when both "location" and "description" are identical.  
3. Your output should be generated in the following format:

<analysis>
Analysis and comparison of each prediction and annotation.
</analysis>

<response>
{
  "matches": [
      {
      'location': the location of the matched object, this should be based on the annotated location,
      'description': your explanation on why you think it is a match.
      },
        {
      'location': ... ,
      'description': ... 
      },
  ]
}
</response>

Be rigorous in considering matches; the location may be slightly differently named, but the description must match overall. 
"""