#!/usr/bin/env bash
# Stop hook：本机有未提交或未推送的工作时提醒，避免换电脑（公司/家里）时另一台拿不到。
# 同一状态只提醒一次（状态记在 .git 里，不进仓库），不是每轮都刷屏；只提示，不阻断。
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
git_dir=$(git rev-parse --git-dir 2>/dev/null) || exit 0
state_file="$git_dir/claude-sync-reminder"

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
branch=${branch//[\"\\]/}
dirty=$(git status --porcelain | wc -l | tr -d ' ')

if git rev-parse --verify --quiet '@{u}' >/dev/null; then
  ahead=$(git rev-list --count '@{u}..HEAD')
  upstream=yes
else
  ahead=0
  upstream=no
fi

if [ "$dirty" = 0 ] && [ "$ahead" = 0 ] && [ "$upstream" = yes ]; then
  rm -f "$state_file"
  exit 0
fi

# 文件数变化不算新状态，否则改代码期间每轮都提醒
state="$branch|$(git rev-parse HEAD)|$([ "$dirty" = 0 ] && echo clean || echo dirty)|$ahead|$upstream"
[ "$(cat "$state_file" 2>/dev/null)" = "$state" ] && exit 0
printf '%s' "$state" > "$state_file"

parts=""
[ "$dirty" != 0 ] && parts="${dirty} 个文件未提交"
if [ "$upstream" = no ]; then
  parts="${parts:+$parts、}整个分支还没推送到 GitHub"
elif [ "$ahead" != 0 ]; then
  parts="${parts:+$parts、}${ahead} 个提交未推送"
fi
printf '{"systemMessage":"[同步提醒] 本机分支 %s：%s。换电脑前记得让 Claude 提交并推送。"}\n' "$branch" "$parts"
exit 0
