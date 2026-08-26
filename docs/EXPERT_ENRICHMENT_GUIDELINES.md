# Expert enrichment guidelines

The feature library combines model-derived candidates with a one-time offline expert enrichment process. Enrichment is performed against MITRE ATT&CK Enterprise v16.1.

## Inputs

1. Positive-SHAP candidate tokens from correctly classified training samples.
2. Official technique/sub-technique name.
3. Definitional phrases in the ATT&CK Description section.
4. Tool identifiers and procedural command strings in Procedure Examples.

## Inclusion rules

- Include a phrase only when it distinguishes the target technique from closely related classes.
- Prefer operational artifacts, command fragments, registry paths, file names, protocol markers, tool identifiers, and behavior-specific verbs.
- Preserve meaningful security tokens and path fragments; normalize only spacing and case variants.
- Record description-derived phrases separately from tool/command-derived entries.
- Add an item to `final_features` only after checking that it is not generic across many ATT&CK classes.

## Exclusion rules

- Exclude generic terms such as `attacker`, `system`, `malware`, or `process` unless they occur inside a discriminative phrase.
- Exclude tactic-level terms that do not distinguish the target technique.
- Exclude phrases inferred from held-out labels, predictions, or test-set error analysis.
- Do not add an unverified co-mentioned technique as a second ground-truth label.

## Quality control

- Keep ATT&CK ID, name, and feature sources traceable in one row.
- Deduplicate case-insensitively while preserving the first spelling.
- Review semantically overlapping neighboring techniques together.
- Freeze the library before final test evaluation; do not tune entries on held-out outcomes.

During peer review, `resources/final_feature_library.csv` and
`resources/final_feature_library.xlsx` contain only release-status notices. The
final feature library will be uploaded after manuscript acceptance.
