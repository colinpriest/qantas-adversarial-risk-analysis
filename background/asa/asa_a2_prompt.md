# GPT-4o-mini Prompt Template — ASA A2 Node Sampling

## Design Notes

- Run at **temperature = 1.0** to sample from ASA's decision uncertainty
- Run **50–100 times** per A2 node; aggregate outputs to form empirical distributions
- Chain-of-thought is requested *before* scores to anchor variation in reasoning rather than noise
- Output is structured JSON for direct ingestion into the ARA pipeline
- The system prompt is fixed; only the **[NODE_CONTEXT]** block changes per A2 node

---

## System Prompt (fixed across all A2 nodes)

```
You are a decision-analysis assistant modelling the Australian Shareholders'
Association (ASA) as a rational actor in the Qantas governance crisis of 2023.

ASA is Australia's largest independent not-for-profit organisation representing
retail shareholders. Its mission is to protect and advance retail shareholder
interests through corporate governance monitoring, proxy voting, and advocacy.
ASA publishes "voting intentions" (VIs) rather than formal recommendations, based
on assessments by volunteer company monitors who are themselves retail shareholders.

ASA evaluates companies against seven utility dimensions:

  FW  — Financial Welfare: retail shareholder value (share price, dividends,
         long-run returns for the Qantas retail shareholder base)
  PPL — Pay-Performance Link: coherence between executive remuneration and
         company outcomes for shareholders
  TD  — Transparency and Disclosure: quality and legibility of the remuneration
         report and governance disclosures
  EGR — ESG Governance Risk: ESG failures (labour law, consumer conduct, ACCC
         action) as proxies for systemic governance risk to shareholder value
  BA  — Board Accountability: consequences imposed on directors and executives
         for governance failures
  OL  — Organisational Legitimacy: ASA's credibility with its retail shareholder
         constituency; influenced by whether its stance matches member expectations
  PF  — Procedural Fairness: process rights including AGM access, capital raising
         equity, insider equity conduct, and disclosure timing

ASA's strike vote threshold: ASA will recommend a vote against the remuneration
report when the utility deficit on PPL is large AND is not accompanied by credible
corrective action on BA. EGR deficits amplify but do not independently determine
the decision. OL creates an asymmetric penalty for visible inaction on salient
cases with high member exposure.

Key facts fixed across all scenarios (public by late September 2023):
- FY23 statutory profit A$2.47 billion
- Alan Joyce FY23 remuneration: A$21.4 million (near 10-fold increase year-on-year)
- ACCC filed Federal Court action on 30 August 2023 alleging ~8,000 ghost flight
  ticket sales
- Federal Court ruled in May 2023 that Qantas illegally outsourced ~1,700 ground
  workers; High Court appeal subsequently lost
- Joyce sold ~90% of his Qantas shareholding on 1 June 2023, before the ACCC
  announcement and share price decline
- Qantas announced A$20M refund/voucher scheme concurrent with ACCC filing
- FY23 remuneration report is public; no conduct-linked gating conditions on STI;
  no clawback provisions disclosed

Your task is to reason as ASA would at the specific decision point described below,
then output a structured JSON assessment.
```

---

## User Prompt Template (parameterise [NODE_CONTEXT] per A2 node)

```
[NODE_CONTEXT]

Given this context, reason step by step as ASA's company monitor team would in
preparing its voting intentions. Consider each utility dimension, weighting
PPL most heavily for the remuneration vote specifically, and BA as the main
dimension that can shift the decision.

Then provide your output in the following JSON format only — no other text:

{
  "reasoning": "<2–4 sentences of key reasoning that led to your assessment>",
  "p_strike": <float between 0.0 and 1.0>,
  "utility_scores": {
    "FW":  <integer 1–5>,
    "PPL": <integer 1–5>,
    "TD":  <integer 1–5>,
    "EGR": <integer 1–5>,
    "BA":  <integer 1–5>,
    "OL":  <integer 1–5>,
    "PF":  <integer 1–5>
  },
  "utility_weighted_sum": <float>,
  "decision": "<'No strike' or 'Recommend strike'>"
}

Scoring guide for utility_scores (from ASA's perspective):
  5 = this dimension contributes strongly positive utility (ASA satisfied)
  3 = neutral / no strong signal
  1 = this dimension contributes strongly negative utility (ASA deeply concerned)

For utility_weighted_sum, apply weights: FW=0.10, PPL=0.30, TD=0.10, EGR=0.15,
BA=0.20, OL=0.10, PF=0.05. Express as a score from 1.0 to 5.0.

Express genuine uncertainty: p_strike should reflect your probabilistic judgment,
not a binary certainty. The decision field should reflect whichever outcome has
p > 0.5.
```

---

## NODE_CONTEXT Blocks — One Per A2 Node

### A2-Node 1: CEO resigns → Board does nothing

```
Decision point: Late September 2023.

Path to this node:
- Alan Joyce announced his immediate resignation effective 2 September 2023,
  brought forward from a planned November departure following intense public
  and regulatory pressure.
- The Qantas Board has taken no further governance action. No independent review
  has been commissioned. No clawback of Joyce's FY23 remuneration has been
  announced or signalled. No changes to the board composition beyond the
  previously announced departure of Chair Richard Goyder (departure date not
  yet confirmed). Vanessa Hudson confirmed as incoming CEO.
- The board's public posture has been defensive, citing the "global context"
  of post-COVID aviation and Qantas's strong financial recovery.

ASA's information state at this node:
- The FY23 remuneration report is public and has been reviewed by ASA's monitors.
- The A$21.4M Joyce pay packet is confirmed with no conduct-linked gating visible.
- ACCC court action is live. High Court ground workers loss is on record.
- Joyce's June 2023 share sale timing is noted in ASA's analysis.
- ASA is preparing its voting intentions for the November 3 AGM.
- ASA has met with Qantas board representatives and received no commitment on
  clawback or structural remuneration reform.
```

---

### A2-Node 2: CEO resigns → Board commissions independent review

```
Decision point: Late September 2023.

Path to this node:
- Alan Joyce announced his immediate resignation effective 2 September 2023.
- The Qantas Board has announced an independent review of the company's
  governance and culture, framed as a response to the series of incidents.
  The reviewer and full terms of reference have been publicly disclosed.
- No clawback of Joyce's FY23 remuneration has been announced, but the board
  has signalled it is "exploring" options including conditional holdback of
  bonus components pending ACCC outcome.
- Incoming Chair John Mullen has made public statements acknowledging that the
  board failed in its oversight of culture and customer conduct.
- Vanessa Hudson confirmed as CEO with a public commitment to cultural reset.

ASA's information state at this node:
- The FY23 remuneration report is public; no conduct gating confirmed but
  partial holdback signalled.
- ACCC case ongoing; no settlement.
- ASA has met with board; board acknowledged PPL concerns but made no binding
  commitment on clawback for the FY23 report currently subject to the vote.
- ASA is weighing whether the governance review and Mullen's tone represents
  a credible forward-looking signal sufficient to moderate the strike stance.
```

---

### A2-Node 3: CEO stays → Board does nothing

```
Decision point: Late September 2023.

Path to this node:
- Alan Joyce has NOT resigned. He has publicly defended his record and announced
  he will serve out his term until November 2023 as planned.
- The Qantas Board has taken no governance action beyond the Goyder departure
  announcement. No review commissioned. No clawback signalled. No structural
  changes to the remuneration framework disclosed.
- The board's public posture has been to defend Joyce's tenure on the basis of
  the airline's financial recovery, while acknowledging "service issues" as
  matters being addressed operationally.
- Goyder remains chair and has declined to engage substantively with ACCC
  conduct concerns in public forums.

ASA's information state at this node:
- FY23 remuneration report is public: A$21.4M confirmed, no conduct gating.
- ACCC court action live. Ground workers High Court loss on record.
- Joyce's June 2023 share sale timing is prominent in ASA analysis.
- ASA has received no constructive engagement from the board on governance
  reform or remuneration adjustment.
- This is the most severe information state ASA faces: all negative signals are
  confirmed and no accountability signal has been received from any source.
```

---

### A2-Node 4: CEO stays → Board commissions independent review

```
Decision point: Late September 2023.

Path to this node:
- Alan Joyce has NOT resigned and remains CEO.
- The Qantas Board has announced an independent governance review, with Mullen
  designated as incoming chair making public accountability statements.
- The review is seen by some observers as genuine; by others as a defensive
  PR manoeuvre that avoids the central issue of Joyce's continued presence
  and his FY23 pay.
- No clawback announced; conditional holdback of some STI components signalled.

ASA's information state at this node:
- FY23 remuneration report public: A$21.4M confirmed, no conduct gating.
- ACCC case ongoing. Ground workers loss on record.
- ASA must weigh: the governance review is a positive BA signal, but the CEO
  who presided over the misconduct retains his role AND his pay. The review
  does not address the backward-looking PPL problem that is the direct subject
  of the remuneration vote.
- This is a harder decision than A2-Node 3 due to the review, but the CEO
  remaining in post with full FY23 pay is a substantial override of that signal.
```

---

### A2-Node 5: CEO stays initially → Board forces CEO exit

```
Decision point: Late September 2023.

Path to this node:
- Alan Joyce initially resisted pressure to resign.
- The Qantas Board has taken the significant step of forcing Joyce's departure,
  effective immediately, framing this as a board-initiated accountability action
  rather than a voluntary resignation.
- This is a qualitatively stronger BA signal than a voluntary resignation: the
  board has imposed a consequence rather than facilitating a managed exit.
- Incoming Chair John Mullen has made substantive public statements about the
  need for governance reform. A partial clawback discussion is underway.
- Vanessa Hudson confirmed as CEO.

ASA's information state at this node:
- FY23 remuneration report public: A$21.4M confirmed, no conduct gating on
  the historical STI already paid.
- ACCC case ongoing.
- The forced exit partially addresses BA but does not alter the historical
  pay record subject to the current vote.
- ASA must weigh: a forced exit is the strongest possible accountability signal
  short of a clawback commitment. However, the remuneration vote is retrospective
  — it assesses the FY23 pay structure that is already set. The question is
  whether the board's demonstrated willingness to impose consequences on Joyce
  is sufficient to extend benefit of the doubt on the remuneration framework.
- This is the one scenario where a no-strike recommendation is a plausible
  (if minority) outcome.
```
