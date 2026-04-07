#!/usr/bin/env bash
#
# Install pre-commit hook to auto-rebuild presentation.pdf
#
# This hook ensures presentation.pdf is always up-to-date with presentation.md
#

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOOK_SOURCE="$REPO_ROOT/slides/scripts/pre-commit-hook.sh"
HOOK_DEST="$REPO_ROOT/.git/hooks/pre-commit"

echo "======================================================================"
echo "INSTALLING GIT PRE-COMMIT HOOK"
echo "======================================================================"
echo ""

# Check if hook source exists
if [ ! -f "$HOOK_SOURCE" ]; then
    echo "ERROR: Hook source not found: $HOOK_SOURCE"
    exit 1
fi

# Backup existing hook if it exists
if [ -f "$HOOK_DEST" ]; then
    BACKUP="$HOOK_DEST.backup.$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing hook to: $BACKUP"
    cp "$HOOK_DEST" "$BACKUP"
fi

# Copy hook
cp "$HOOK_SOURCE" "$HOOK_DEST"
chmod 755 "$HOOK_DEST"

echo "✓ Hook installed: $HOOK_DEST"
echo ""
echo "======================================================================"
echo "WHAT THIS DOES"
echo "======================================================================"
echo ""
echo "The pre-commit hook will:"
echo "  1. Detect changes to slides/presentation.md"
echo "  2. Rebuild slides/presentation.pdf automatically"
echo "  3. Stage the updated PDF for commit"
echo ""
echo "Benefits:"
echo "  - Never commit stale PDFs"
echo "  - PDF always matches source"
echo "  - No manual rebuild needed"
echo ""
echo "To bypass (not recommended):"
echo "  git commit --no-verify"
echo ""
echo "======================================================================"
echo "INSTALLED SUCCESSFULLY"
echo "======================================================================"
