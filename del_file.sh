rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now
