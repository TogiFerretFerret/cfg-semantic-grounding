#!/usr/bin/env bash
# Run with: sudo bash fix_sweagent_perms.sh
mkdir -p /root/tools
chown -R river:river /root/tools
chmod 755 /root/tools
# sweagent also writes these files
touch /root/.swe-agent-env /root/state.json
chown river:river /root/.swe-agent-env /root/state.json
echo "Done. /root/tools is now writable by river."
