#!/usr/bin/env bash

# ==============================================================================
# SECURE AND ISOLATED CCS (CLAUDE CODE SWITCH) SETUP SCRIPT
# ==============================================================================
# Target: Only the current user ('iec') can access and execute CCS/Claude.
# Directory: Completely isolated inside '/home/iec/Parallel_Computing_on_FPGA'.
# Config & Auth: Contained in a secure '.ccs_local' directory.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Define directories
PROJECT_DIR="/home/iec/Parallel_Computing_on_FPGA"
CCS_LOCAL_DIR="${PROJECT_DIR}/.ccs_local"
CCS_CONFIG_DIR="${CCS_LOCAL_DIR}/ccs_config"
CLAUDE_CONFIG_DIR="${CCS_LOCAL_DIR}/claude_config"

echo "=== 🛡️ Securing Parallel_Computing_on_FPGA Permissions ==="
# Step 1: Force strict user-only permissions (700) on the project directory.
# This prevents any other UNIX users from reading, writing, or traversing this folder.
chmod 700 "${PROJECT_DIR}"
echo "✓ Project directory permissions secured (chmod 700)."

# Step 2: Create isolated config and package directories
echo "=== 📁 Creating Private Configuration Directories ==="
mkdir -p "${CCS_LOCAL_DIR}"
mkdir -p "${CCS_CONFIG_DIR}"
mkdir -p "${CLAUDE_CONFIG_DIR}"

# Force 700 permissions on all environment files to be absolutely bulletproof
chmod -R 700 "${CCS_LOCAL_DIR}"
echo "✓ Isolated environment directories created & secured."

# Step 3: Install @kaitranntt/ccs & @anthropic-ai/claude-code locally
echo "=== 📦 Installing CCS & Claude Code (Isolated) ==="
if [ ! -f "${CCS_LOCAL_DIR}/package.json" ]; then
    echo "Initializing local npm environment with custom package.json..."
    echo '{"name": "ccs-local-env", "version": "1.0.0", "private": true}' > "${CCS_LOCAL_DIR}/package.json"
    
    echo "Installing @kaitranntt/ccs and @anthropic-ai/claude-code locally..."
    cd "${CCS_LOCAL_DIR}"
    npm install @kaitranntt/ccs @anthropic-ai/claude-code
else
    echo "✓ Local npm environment already exists. Upgrading/verifying packages..."
    cd "${CCS_LOCAL_DIR}"
    npm install @kaitranntt/ccs @anthropic-ai/claude-code
fi
echo "✓ Installation completed successfully inside .ccs_local."

# Step 4: Write Wrapper Scripts
echo "=== ⚙️ Creating Wrapper Scripts ==="

# Write CCS wrapper
cat << 'EOF' > "${PROJECT_DIR}/ccs"
#!/usr/bin/env bash

# Force absolute directories for isolation
PROJECT_DIR="/home/iec/Parallel_Computing_on_FPGA"
export CCS_DIR="${PROJECT_DIR}/.ccs_local/ccs_config"
export CLAUDE_CONFIG_DIR="${PROJECT_DIR}/.ccs_local/claude_config"

CCS_BIN="${PROJECT_DIR}/.ccs_local/node_modules/.bin/ccs"

if [ ! -f "$CCS_BIN" ]; then
    echo "❌ Error: CCS binary not found. Please run ./setup_ccs.sh first."
    exit 1
fi

# Execute the local CCS binary
exec "$CCS_BIN" "$@"
EOF

# Write Claude Code wrapper
cat << 'EOF' > "${PROJECT_DIR}/claude"
#!/usr/bin/env bash

# Force absolute directories for isolation
PROJECT_DIR="/home/iec/Parallel_Computing_on_FPGA"
export CCS_DIR="${PROJECT_DIR}/.ccs_local/ccs_config"
export CLAUDE_CONFIG_DIR="${PROJECT_DIR}/.ccs_local/claude_config"

CLAUDE_BIN="${PROJECT_DIR}/.ccs_local/node_modules/.bin/claude"

if [ ! -f "$CLAUDE_BIN" ]; then
    echo "❌ Error: Claude Code binary not found. Please run ./setup_ccs.sh first."
    exit 1
fi

# Execute the local Claude Code binary
exec "$CLAUDE_BIN" "$@"
EOF

# Set wrappers executable and owner-only
chmod 700 "${PROJECT_DIR}/ccs"
chmod 700 "${PROJECT_DIR}/claude"
echo "✓ Created ./ccs and ./claude executable wrappers (chmod 700)."

# Step 5: Update .gitignore to avoid committing credentials or heavy packages
echo "=== 🔒 Updating .gitignore ==="
GITIGNORE_FILE="${PROJECT_DIR}/.gitignore"
if [ -f "$GITIGNORE_FILE" ]; then
    if ! grep -q ".ccs_local" "$GITIGNORE_FILE"; then
        echo -e "\n# Claude Code Switch & Claude local configs\n.ccs_local/\nccs\nclaude" >> "$GITIGNORE_FILE"
        echo "✓ Appended isolated files to .gitignore."
    else
        echo "✓ .ccs_local already ignored in .gitignore."
    fi
else
    echo -e ".ccs_local/\nccs\nclaude" > "$GITIGNORE_FILE"
    chmod 600 "$GITIGNORE_FILE"
    echo "✓ Created .gitignore with isolated directories."
fi

echo "=============================================================================="
echo "🎉 SETUP COMPLETED SUCCESSFULLY!"
echo "=============================================================================="
echo "🔒 Your ccs and claude setups are completely isolated."
echo "🚫 Other UNIX users CANNOT read or write your configs or credentials."
echo "🚀 To run Claude Code Switch:  ./ccs"
echo "🚀 To run Claude Code directly: ./claude"
echo "💡 Tip: You can add these aliases to your ~/.zshrc or ~/.bashrc for convenience:"
echo "   alias ccs=\"${PROJECT_DIR}/ccs\""
echo "   alias claude=\"${PROJECT_DIR}/claude\""
echo "=============================================================================="
