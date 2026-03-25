## Scope
- Keep changes focused on the `vocoder/` package, `tests/`, packaging metadata,
  and top-level documentation.
- Treat generated audio files, archived dependency snapshots, and example data
  as support material unless the task is explicitly about them.
- Ignore editor swap files such as `.swp` unless the task is explicitly about
  cleaning them up.
- Prefer small, direct refactors over wide rewrites.
- Run git operations sequentially. Do not run `git add`, `git commit`, or
  `git push` concurrently, because that can leave a stale `.git/index.lock`.

## Code Style
- Match the existing style of the file you are editing, when in doubt use
  the style guides below as a reference.
- Use multi-line formatting for function definitions and function calls when
  they exceed the line limit.
- Target a maximum line length of 80 characters for new or modified lines when
  practical.
- Indent continued arguments by one level (4 spaces).
- Keep the closing parenthesis on the same line as the last argument.
- Do not vertically align arguments.
- Use single quotes unless the surrounding code clearly uses double quotes.
- Keep functions small and direct.
- Prefer explicit code over clever abstractions.
- Avoid unrelated refactors while making a requested change.
- Keep docstrings short and practical.
- For public or non-trivial functions, prefer the existing triple-single-quote
  docstring style used in this repo.
- Document parameters only when the signature or behavior is not obvious.
- When a parameter list is useful, keep it compact, for example:
  '''Short explanation.
  parameter_name: description
  '''
- Preserve the current public API unless the task explicitly calls for a change.
