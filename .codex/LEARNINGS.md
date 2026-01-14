# Learnings

- Initialized learnings log.
- Signpost: When encountering unexpected file type changes, confirm with the user before committing (AGENTS.md symlink update).
- Friction: pre-commit fixed trailing whitespace and missing EOF in `AGENTS.md`; re-stage the file before committing.
- Friction: `gh pr merge --delete-branch` failed because `main` is checked out in the bd worktree; merge succeeded but local branch cleanup needs manual handling.
- Friction: cannot check out `main` in the bd worktree while `main` is already checked out in the primary worktree; need to move primary off `main` or keep bd worktree detached.
- Friction: avoid backticks in `bd comments add ... "..."` strings (zsh command substitution); use single quotes or escape backticks.
- Signpost: Gemini CLI model name for “3 Pro Preview” is `gemini-3-pro-preview` (using `gemini-3-pro` or missing/incorrect model names can yield “Requested entity was not found”).
- Friction: running `gemini --yolo` inside the repo can modify the working tree; run Gemini from a temp directory when you only want advisory output, or avoid YOLO so tool calls require approval.
