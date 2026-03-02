# Data in `ranked_voting_recommendations.csv`

The `voting_diff` column represents the **year-on-year change in shareholder dissatisfaction**.

It measures the difference between the percentage of votes cast *“Against”* a resolution (typically the Remuneration Report) in the current year compared to the previous year.

---

## The Formula

$$
\text{voting\_diff} = \text{rem\_against\_pct} - \text{prior\_year\_pct}
$$

---

## How to Interpret the Numbers

- **Positive Number (e.g., +0.30):**  
This indicates an **escalation in protest**. It means the *“Against”* vote increased by 30 percentage points compared to last year. This usually happens when a *Headline Incident* occurs or when shareholders feel the board has not addressed previous concerns.
- **Negative Number (e.g., -0.15):**  
This indicates a **“cooling off” or improvement**. It means the *“Against”* vote dropped by 15 percentage points, suggesting that board actions or improved performance have successfully regained some shareholder trust.
- **Zero:**  
The level of dissent remained exactly the same as the previous year.

---

## Why It Was Used in Your Analysis

In the extended data you provided, `voting_diff` was used to test whether **Board Corrective Actions** actually work.

- **Rank 1 actions (Announcements / Reviews)** resulted in the lowest `voting_diff` (+14.9%), suggesting these actions are the most effective at *dampening* the growth of shareholder anger.
- **Rank 2 actions (Sackings)** often had very high positive `voting_diff` values (+46.7%), because the sackings were usually a late response to a massive crisis that was already driving a large spike in *“Against”* votes.

