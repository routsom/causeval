# B2 RAG causal grounding (live)

- **app / judge model**: claude-haiku-4-5-20251001 (real Claude RAG app; deterministic outcome functions)
- **generated_at**: 2026-09-30T20:35:28.961555+00:00
- **n_items**: 12 (6 fictional, 6 well-known)   **repeats**: 3

**Counterfactual Adherence AUROC (fictional vs well-known): 0.833**

RAG grounding (tau=0.5, policy=follow_context)
  context_reliance: +0.500 [95% CI +0.174, +0.826] (n=12, R=3)
  counterfactual_adherence: +0.722 [95% CI +0.438, +0.957] (n=12, R=3)
  classes: grounded=8, parametric=4

| item | group | correct(full) | correct(none) | Context Reliance | Counterfactual Adherence | class |
|---|---|---|---|---|---|---|
| f0 | fictional | 0.67 | 0.00 | +0.67 | 1.00 | grounded |
| f1 | fictional | 0.33 | 0.00 | +0.33 | 1.00 | grounded |
| f2 | fictional | 1.00 | 0.00 | +1.00 | 1.00 | grounded |
| f3 | fictional | 1.00 | 0.00 | +1.00 | 1.00 | grounded |
| f4 | fictional | 1.00 | 0.00 | +1.00 | 1.00 | grounded |
| f5 | fictional | 1.00 | 0.00 | +1.00 | 1.00 | grounded |
| w0 | well-known | 1.00 | 1.00 | +0.00 | 0.33 | parametric |
| w1 | well-known | 1.00 | 1.00 | +0.00 | 0.00 | parametric |
| w2 | well-known | 1.00 | 1.00 | +0.00 | 1.00 | grounded |
| w3 | well-known | 1.00 | 1.00 | +0.00 | 0.00 | parametric |
| w4 | well-known | 1.00 | 1.00 | +0.00 | 0.33 | parametric |
| w5 | well-known | 1.00 | 0.00 | +1.00 | 1.00 | grounded |

> CA edits one supporting fact to a false value and checks whether the answer follows it. High CA = the answer causally depends on the context (grounded); low CA = the model answered from its parametric prior. DeepEval Faithfulness would call the edited-context answer 'faithful' in both cases, so it cannot make this distinction.
