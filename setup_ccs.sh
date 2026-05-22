#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
CCS_LOCAL_DIR="${PROJECT_DIR}/.ccs_local"
CCS_CONFIG_DIR="${CCS_LOCAL_DIR}/ccs_config"
CLAUDE_CONFIG_DIR="${CCS_LOCAL_DIR}/claude_config"

echo "=== Securing Parallel_Computing_on_FPGA permissions ==="
chmod 700 "${PROJECT_DIR}"
echo "Project directory permissions secured."

echo "=== Creating private configuration directories ==="
mkdir -p "${CCS_CONFIG_DIR}" "${CLAUDE_CONFIG_DIR}"
chmod -R 700 "${CCS_LOCAL_DIR}"
echo "Isolated environment directories created."

echo "=== Installing CCS and Claude Code locally ==="
if [ ! -f "${CCS_LOCAL_DIR}/package.json" ]; then
    printf '%s\n' '{"name":"ccs-local-env","version":"1.0.0","private":true}' > "${CCS_LOCAL_DIR}/package.json"
fi

cd "${CCS_LOCAL_DIR}"
npm install @kaitranntt/ccs @anthropic-ai/claude-code
echo "Installation completed inside .ccs_local."

echo "=== Creating wrapper scripts ==="

cat << 'EOF' > "${PROJECT_DIR}/ccs"
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
export CCS_DIR="${PROJECT_DIR}/.ccs_local/ccs_config"
export CLAUDE_CONFIG_DIR="${PROJECT_DIR}/.ccs_local/claude_config"

CCS_BIN="${PROJECT_DIR}/.ccs_local/node_modules/.bin/ccs"

if [ ! -f "${CCS_BIN}" ]; then
    echo "Error: CCS binary not found. Run ./setup_ccs.sh first."
    exit 1
fi

exec "${CCS_BIN}" "$@"
EOF

cat << 'EOF' > "${PROJECT_DIR}/claude"
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
export CCS_DIR="${PROJECT_DIR}/.ccs_local/ccs_config"
export CLAUDE_CONFIG_DIR="${PROJECT_DIR}/.ccs_local/claude_config"

CLAUDE_BIN="${PROJECT_DIR}/.ccs_local/node_modules/.bin/claude"

if [ ! -f "${CLAUDE_BIN}" ]; then
    echo "Error: Claude Code binary not found. Run ./setup_ccs.sh first."
    exit 1
fi

exec "${CLAUDE_BIN}" "$@"
EOF

chmod 700 "${PROJECT_DIR}/ccs" "${PROJECT_DIR}/claude"
echo "Created ./ccs and ./claude wrappers."

echo "=== Updating .gitignore ==="
GITIGNORE_FILE="${PROJECT_DIR}/.gitignore"
if [ -f "${GITIGNORE_FILE}" ]; then
    if ! grep -q "^\.ccs_local/" "${GITIGNORE_FILE}"; then
        printf '\n# Claude Code Switch and Claude local configs\n.ccs_local/\nccs\nclaude\n' >> "${GITIGNORE_FILE}"
        echo "Appended isolated files to .gitignore."
    else
        echo ".ccs_local already ignored in .gitignore."
    fi
else
    printf '.ccs_local/\nccs\nclaude\n' > "${GITIGNORE_FILE}"
    chmod 600 "${GITIGNORE_FILE}"
    echo "Created .gitignore with isolated directories."
fi

echo "=============================================================================="
echo "Setup completed successfully."
echo "CCS and Claude configs are isolated under ${CCS_LOCAL_DIR}."
echo "To run Claude Code Switch: ./ccs"
echo "To run Claude Code directly: ./claude"
echo "Suggested aliases:"
echo "  alias ccs=\"${PROJECT_DIR}/ccs\""
echo "  alias claude=\"${PROJECT_DIR}/claude\""
echo "=============================================================================="
