C0 (initialisation only — no actions)
  State drawn/loaded:
    B0_mkt ~ checkpoint posterior
    B0_mgmt ~ checkpoint posterior (or derived)
    V_deference fixed
    Flags: review_started=0, ceo_removed=0

  |
  v

C1 (Oct 2023) — BOARD MOVES: choose D1_action ∈ {D0, D1, D2}
  |
  +-- D0: Do nothing (status quo)
  |     State: review_started=0, ceo_removed=0
  |     -> proceed to C2
  |
  +-- D1: Commission review (ONE-TIME)
  |     State update:
  |       review_started=1
  |       latent review severity R ~ Prior(R | state at C1)
  |     -> proceed to C2
  |
  +-- D2: Sack CEO immediately (ONE-TIME)
        State update:
          ceo_removed=1
          ceo_mode = "sacked"
        -> proceed to C2

  |
  v

C2 (pre-AGM 2023) — ASA MOVES: choose A2 ∈ {DoNothing, RecommendStrike}
  |
  +-- DoNothing
  |     ASA_shift = 0
  |     -> proceed to C3a
  |
  +-- RecommendStrike
        ASA_shift = gamma_ASA_rec_strike_vote_logit
        -> proceed to C3a

  |
  v

C3a (AGM 2023 outcome) — NO CHOICES (stochastic outcome node)
  Realise continuous vote:
    V_2023 = rem_against_pct ∈ [0,1]
  Derived (reporting/diagnostics):
    strike1 = 1[V_2023 > 0.25]
    category:
      cat0: V_2023 ≤ 0.25
      cat1: 0.25 < V_2023 ≤ τ_overwhelm
      cat2: V_2023 > τ_overwhelm  (optional; diagnostic only)
  -> proceed to C3b

  |
  v

C3b (2024 pre-release; review nearly complete) — CEO MAY PRE-EMPT
  Condition: only if ceo_removed=0
  CEO chooses M_2024 ∈ {Resign, Stay}
  |
  +-- If ceo_removed=1 already (from C1 D2):
  |       (skip CEO choice)
  |       -> proceed to C4 with CEO absent
  |
  +-- If CEO Resigns:
  |       State update:
  |         ceo_removed=1
  |         ceo_mode = "resigned"
  |       -> proceed to C4
  |
  +-- If CEO Stays:
          State unchanged (ceo_removed=0)
          -> proceed to C4

  |
  v

C4 (2024 review released / board response) — BOARD MOVES (if CEO still present)
  Condition: if ceo_removed=0
  Board chooses D4_action ∈ {D0, D2}
  |
  +-- If CEO already removed (resigned or sacked):
  |       Board action is moot regarding CEO removal
  |       (optionally: governance clean-up actions, outside current scope)
  |       -> end / proceed to next cycle modelling
  |
  +-- If D0: Do nothing further
  |       -> end / proceed to next cycle
  |
  +-- If D2: Sack CEO (ONE-TIME)
          State update:
            ceo_removed=1
            ceo_mode = "sacked"
          -> end / proceed to next cycle