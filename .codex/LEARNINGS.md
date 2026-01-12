# Learnings

- Initialized learnings log.
- Signpost: When encountering unexpected file type changes, confirm with the user before committing (AGENTS.md symlink update).
- Friction: pre-commit fixed trailing whitespace and missing EOF in `AGENTS.md`; re-stage the file before committing.
- Friction: `gh pr merge --delete-branch` failed because `main` is checked out in the bd worktree; merge succeeded but local branch cleanup needs manual handling.
- Friction: cannot check out `main` in the bd worktree while `main` is already checked out in the primary worktree; need to move primary off `main` or keep bd worktree detached.
