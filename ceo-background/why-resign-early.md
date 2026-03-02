# CEO Utility For Resigning Immediately

Alan Joyce’s bonus outcomes were not determined by a specific “AGM vote on his bonus” that he could avoid by resigning early, because:

- At the 2023 AGM, shareholders voted on the **remuneration report**, not directly on whether Joyce personally kept or lost his bonus. The chair explicitly acknowledged an “overwhelming vote against our remuneration report,” i.e., a record protest strike, but that vote is advisory under Australian law.investor.qantas+1
- Joyce’s FY23 remuneration was instead controlled by the **board’s discretion** and later by the governance review. The board initially reserved the right to withhold up to A$14.4m of his variable pay, and after the governance review it decided to cut his FY23 remuneration by A$9.26m (forfeiting long‑term incentives and reducing short‑term incentives), independent of any AGM [timing.abc](http://timing.abc)+5

So while resigning before the AGM may have been politically convenient, the loss of a large portion of his bonus ultimately came from the board’s post‑review decision, not from an AGM vote he sidestepped.

You can’t know his true utility function, but you can frame “leave now vs stay” using what’s on the record plus generic board‑behaviour economics.

## 1. Stated and widely inferred reasons

Publicly, Qantas and Joyce framed the early exit as about **accelerating renewal** and responding to pressure:

- Qantas said he would bring forward retirement “to help the company accelerate its renewal,” and Joyce said the focus on Qantas and events of the past made it clear the company needed to move ahead with renewal as a [priority.abc](http://priority.abc)+2
- Commentators described the move as bowing to intense public and political anger following the ACCC lawsuit over cancelled flights and broader controversies.aerotime+4

So at a minimum, an immediate resignation reduces ongoing reputational damage to the company and the board, and potentially protects his own public image relative to being forced out after a bruising AGM.

## 2. Bonus and pay dynamics

The bonus story cuts both ways:

- At the time of his September 2023 exit it was already unclear “what will happen with his entitlements now he has stepped down”; unions and others were calling for his bonuses to be stripped.wsws+1
- The later governance review found “considerable harm” from leadership decisions and led the board in August 2024 to cut more than A$9.2m from his final payout, including forfeited long‑term incentives and reduced short‑term incentives, even while finding no deliberate [wrongdoing.ch](http://wrongdoing.ch)-aviation+7

That outcome shows: resigning early did **not** protect him from significant clawback once the board had a governance review in hand. However, from Joyce’s perspective in early September 2023, stepping down might plausibly have increased the probability of retaining *some* bonus compared with waiting to be pushed out after further reputational damage and a hostile AGM.

In a utility story for your case study, you can reasonably posit:

- If he stayed through a disastrous AGM, the political pressure on the board to zero out his variable pay ex post would have been even stronger.
- By exiting “voluntarily” and giving the board a clean slate, he improves the optics of partial, rather than total, forfeiture.

You can’t prove that with data, but it is behaviourally consistent with how boards balance accountability and contractual expectations.

## 3. Other utilities from leaving immediately

Beyond bonus protection, there are several tangible and intangible benefits to leaving when he did:

- **Avoiding a public humiliation at the AGM:** Shareholder anger over pay and governance was already intense; the 2023 AGM subsequently delivered an overwhelming vote against the remuneration report. Exiting beforehand spared him direct confrontation and questioning in that forum.investor.qantas+1
- **Protecting the board and institution:** Early removal of a lightning‑rod CEO is a classic damage‑control tool: it gives the board something visible to offer angry stakeholders and regulators and supports the “new management, new culture” narrative.reuters+4
- **Legal and regulatory posture:** Stepping aside while ACCC and other proceedings were live separates ongoing defence and remediation from the individual most associated with the alleged conduct, which can be useful in settlement dynamics and in board‑commissioned reviews.flightglobal+2
- **Personal career/legacy:** He exits with a record profit and can frame his departure as putting the company first, rather than being pushed after a formal finding of wrongdoing. That has non‑trivial value for future roles, boards, speaking, and reputation.fortune+2

## 4. How to use this in your model

For your case study utility comparison:

- Model his choice as trading off:
  - foregone salary for two months,
  - a higher chance that *some* bonus / LTI is preserved,
  - reduced probability of more severe reputational and legal fallout attaching personally to him,
  - increased probability that the board and review frame problems as “systemic” rather than “this CEO refused to go.”
- Model the board’s side as gaining:
  - a salient accountability signal (CEO exit under pressure),
  - a stronger mandate to commission a hard governance review,
  - better odds of calming investors, regulators and politicians.

Your specific hypothesis—early resignation as a **signal** that makes it politically easier for the board to leave him with *some* bonus rather than none—is not directly documented in the reporting, but it is entirely consistent with the subsequent pattern: big cuts and forfeitures, but not a full wipe‑out, after a review that stressed “no deliberate wrongdoing” alongside “considerable harm.”abc+6

# Bayesian Assumptions

You can model this compactly with a two‑state world (resign now vs stay and be sacked), and then put a simple parametric utility over money plus a big non‑monetary “punishment” term.

## 1. Monetary part: CRRA utility over wealth

Let W*W* be Joyce’s eventual personal wealth from Qantas (salary over next months + expected bonus / LTI actually received, net of legal costs and clawbacks). Use constant‑relative‑risk‑aversion (CRRA):

Umoney(W)={W1−γ1−γγ≠1ln⁡Wγ=1*U*money(*W*)={1−*γW*1−*γ*ln*Wγ*=1*γ*=1

A reasonable **Bayesian prior** on the risk‑aversion parameter for a very wealthy executive could be:

- γ∼N(1.5, 0.52)*γ*∼N(1.5,0.52) truncated to [0.5,3][0.5,3].

That centres him as moderately risk‑averse but not extremely so.

## 2. Non‑monetary penalty: reputation / legal disutility

Let D*D* be an additive disutility term capturing reputation damage, public humiliation (e.g., at AGM), and being a named defendant in legal action:

Utotal=Umoney(W)−D*U*total=*U*money(*W*)−*D*

Model D*D* differently by scenario:

- If he **resigns now**: DR∼N(μR, σR2)*DR*∼N(*μR*,*σR*2)
  - Prior: μR≈50*μR*≈50, σR≈20*σR*≈20 (moderate stigma but framed as “taking responsibility”).
- If he **stays and is sacked + drawn into legal proceedings**: DS∼N(μS, σS2)*DS*∼N(*μS*,*σS*2)
  - Prior: μS≈150*μS*≈150, σS≈40*σS*≈40 (three times the reputational disutility, more uncertainty).

These are in the same arbitrary “utils” scale as Umoney*U*money; you’d calibrate them so that the expected penalty is of similar magnitude to a meaningful fraction of his financial utility (e.g., equivalent to losing many millions).

## 3. Money outcomes by scenario

Let:

- WR*WR*: wealth if he resigns now
- WS*WS*: wealth if he stays and is sacked, **given** your assumption that sacking and legal action are certain.

You could encode priors like:

- WR∼N(μWR, σWR2)*WR*∼N(*μWR*,*σWR*2) with
  - μWR≈8*μWR*≈8 million, σWR≈2*σWR*≈2 million (some pay, some bonus forfeited).
- WS∼N(μWS, σWS2)*WS*∼N(*μWS*,*σWS*2) with
  - μWS≈4*μWS*≈4 million, σWS≈3*σWS*≈3 million (higher chance of severe clawback and higher legal costs, more dispersion).

These priors encode that, ex ante, staying exposes him to both **lower mean** and **higher variance** of financial outcomes once the board and courts respond to a very public sacking.

## 4. Bayesian structure

You then have:

- Prior on γ*γ*, the risk‑aversion parameter.
- Priors on WR,WS*WR*,*WS* and on DR,DS*DR*,*DS*.

In Bayesian terms, you could write his *scenario‑specific* expected utilities as:

EU(resign)=E[Umoney(WR)−DR ∣ information at Sep 2023]*EU*(resign)=E[*U*money(*WR*)−*DR* information at Sep 2023]EU(stay)=E[Umoney(WS)−DS ∣ information at Sep 2023]*EU*(stay)=E[*U*money(*WS*)−*DS* information at Sep 2023]

Given your assumption of **certainty** of sacking and legal action if he stays, there’s no scenario probability on that branch: the uncertainty is only over the magnitudes.

## 5. Interpretation

With the parameterisations above:

- CRRA with γ*γ* around 1.5 makes him dislike the higher variance in WS*WS*.
- Lower mean WS*WS* and much larger DS*DS* make EU(stay)*EU*(stay) substantially lower than EU(resign)*EU*(resign) for most draws from the priors.

That gives you a clean, explicit Bayesian assumption set for your case write‑up, and you can tweak μ*μ* and σ*σ* values in W*W* and D*D* to fit whatever numerical story you want to tell.