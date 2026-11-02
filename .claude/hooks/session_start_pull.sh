#!/usr/bin/env bash
# SessionStart hook：开工时把 develop 快进到 origin/develop，让公司/家里两台电脑从同一份代码开始。
# 只做 fast-forward：不合并、不碰未提交改动、不切分支；任何失败都只提示，不阻断会话。
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# 消息里不能有双引号/反斜杠（手拼 JSON，Windows 上没有 jq）
emit() {
  printf '{"systemMessage":"%s","hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"%s"}}\n' "$1" "$1"
}

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
branch=${branch//[\"\\]/}
if [ "$branch" != "develop" ]; then
  emit "[同步] 当前在分支 ${branch}，不是 develop，未自动拉取。"
  exit 0
fi
if [ -n "$(git status --porcelain)" ]; then
  emit "[同步] 工作区有未提交改动，未自动拉取；处理完后再 git pull --ff-only。"
  exit 0
fi
if ! git fetch --quiet origin develop 2>/dev/null; then
  emit "[同步] 连不上 origin，未拉取（离线？）。"
  exit 0
fi

behind=$(git rev-list --count HEAD..origin/develop)
ahead=$(git rev-list --count origin/develop..HEAD)
if [ "$behind" = 0 ]; then
  if [ "$ahead" = 0 ]; then
    emit "[同步] develop 已是最新。"
  else
    emit "[同步] develop 已是最新，但本地有 ${ahead} 个提交未推送。"
  fi
  exit 0
fi
if [ "$ahead" != 0 ]; then
  emit "[同步] 本地与 origin/develop 已分叉（本地 ${ahead} 个 / 远端 ${behind} 个），未自动合并，需要手动处理。"
  exit 0
fi
if git merge --ff-only --quiet origin/develop 2>/dev/null; then
  emit "[同步] 已从 origin/develop 快进拉取 ${behind} 个提交。"
else
  emit "[同步] 快进拉取失败，需要手动处理。"
fi
exit 0
