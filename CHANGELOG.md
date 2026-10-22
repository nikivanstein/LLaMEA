# Changelog

## 1.4.1 (2026-10-22)

Bug-fix release.

### Fixes
- **Operator prompts were swapped with their names** (#151). `LLaMEA` built the
  default operator, and operators given as plain strings, with positional
  arguments in the wrong order, so the LLM received the literal word
  `"mutation"` instead of the mutation prompt. Operators are now built with
  keyword arguments.
- **A single operator is accepted** (#151). Passing one `str` or `Operator` as
  `operators` no longer splits a string into one operator per character.
- **`Solution.operator_id` is kept** (#151). It is now set in `__init__`,
  included in `to_dict()` and `copy()`, and survives pickling (older pickles
  default it).
- **Crossover example and docs** (#151). `examples/crossover.py` and
  `docs/crossover.rst` used the wrong `Operator` argument order, class name and
  had a syntax error.

### Added
- `Anthropic_LLM` wrapper for Claude models (#149).

### Dependencies
- `werkzeug` 3.1.8 → 3.1.9 (#150).
