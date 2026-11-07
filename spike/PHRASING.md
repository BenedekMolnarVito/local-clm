# Phrasing rules for local CLM questions (P1.3)

The CLM heads compare the state embedding with one embedding per option. Options must differ in meaning, and they must read like prose.

1. Give every yes/no question `criteria: {"true": "...", "false": "..."}`.
2. Write the true and false descriptions as two full sentences that describe two different situations. Do not write one sentence and its negation.
3. Keep the original `instructions` text unchanged.
4. Write each choice option description as one prose sentence that says when the option fits. Keep the option keys unchanged.
5. Write each score level as one full sentence that describes a task at that level. Keep the number and the order of the levels.
6. Write state values as plain prose or plain text lines. Do not put JSON strings inside values. Keep the facts of the state unchanged.
7. Do not add facts that give away the answer.

## Examples

Yes/no (bash-guard):
```json
"criteria": {
  "true": "The command deletes or overwrites source code, commits, history or data that cannot be rebuilt from other files.",
  "false": "The command only removes build output, caches, dependencies or temporary files that the tools can rebuild."
}
```

Choice (category):
```json
"criteria": {
  "bug": "The report describes existing behavior that is broken or gives a wrong result.",
  "feature": "The report asks for new behavior that the product does not have yet.",
  "docs": "The report says the documentation is wrong, missing or out of date."
}
```

Score (complexity):
```json
"criteria": [
  "The task is a trivial or mechanical edit, such as a rename, a typo fix, a dependency bump or a one-line change.",
  "The task is a scoped implementation or edit with clear acceptance criteria.",
  "The task needs design, root-cause debugging, a multi-file refactor or reasoning about an ambiguous goal."
]
```
