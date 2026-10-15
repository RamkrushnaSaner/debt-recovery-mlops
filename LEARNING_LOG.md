# Learning log

## Day 1 - Git and GitHub

What I built: a public repo with a protected main branch, the project
skeleton, params.yaml, and a README, all merged through pull requests.

What I learned:
- working folder -> git add -> git commit -> git push
- branch -> pull request -> squash merge keeps main a straight line
- a ruleset blocks direct pushes, even mine
- rebase replays my commit on top of the new main; resolve, git add, continue

Problems I hit and how I fixed them:
- Commands ran in PowerShell instead of Git Bash: check the prompt (PS vs MINGW64).
- "[200~" paste glitch: press Ctrl+C and retype the line.
- A file only exists after saving: the tab must have no dot.
- Pressing Enter in YAML auto-indented a blank line: read git diff before every commit.
- A merge conflict: git rebase origin/main, fix the markers, push --force-with-lease.
