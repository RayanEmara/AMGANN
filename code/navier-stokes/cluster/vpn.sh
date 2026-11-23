#!/usr/bin/env bash
# Run locally. Both clients use the department portal.
set -eu
portal=gp-dmat-saml.vpn.polimi.it
case "${1:-}:${2:-}" in
    cli:connect|cli:disconnect)
        cli=$(command -v gpclient) || { echo "Install gpclient first (see CLUSTER.md)." >&2; exit 1; }
        if [[ $2 == connect ]]; then
            exec sudo "$cli" connect "$portal" --browser --auto-gateway
        fi
        exec sudo "$cli" disconnect ;;
    paloalto:connect|paloalto:disconnect)
        echo "In GlobalProtect, select $portal and click $2."
        if [[ $(uname -s) == Darwin ]]; then
            exec open -a GlobalProtect
        fi
        echo 'Open the installed GlobalProtect application on your workstation.' ;;
    *) echo "Usage: bash $0 {cli|paloalto} {connect|disconnect}" >&2; exit 2 ;;
esac
