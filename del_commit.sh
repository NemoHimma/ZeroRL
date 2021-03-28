git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch DQN/DQN1/' --prune-empty --tag-name-filter cat -- --all
git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch DQN/DQN4/' --prune-empty --tag-name-filter cat -- --all
