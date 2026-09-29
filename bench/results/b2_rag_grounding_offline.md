# B2 RAG grounding (offline synthetic)

- **generated_at**: 2026-09-29T14:59:38.023470+00:00
- **n_items**: 40
- **repeats**: 8
- **note**: Synthetic behaviour-driven app; not a real judge.

**Counterfactual Adherence AUROC (grounded vs parametric): 1.000**

RAG grounding (tau=0.5, policy=follow_context)
  context_reliance: +0.463 [95% CI +0.311, +0.621] (n=40, R=8)
  counterfactual_adherence: +0.506 [95% CI +0.372, +0.643] (n=40, R=8)
  classes: grounded=20, parametric=20

> CA separates items that causally use the context from items answering parametrically. B2 target: AUROC >= 0.9.