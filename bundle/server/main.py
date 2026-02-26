#!/usr/bin/env python3
"""Entry point for the zvec-mcp MCPB bundle.

This script bootstraps the FastMCP server. When running inside
an MCPB bundle, PYTHONPATH is set to ${__dirname}/server/lib so
all bundled dependencies are importable.
"""

import sys
import os

# Ensure the bundled lib directory is on sys.path
bundle_lib = os.path.join(os.path.dirname(__file__), "lib")
if os.path.isdir(bundle_lib) and bundle_lib not in sys.path:
    sys.path.insert(0, bundle_lib)

from zvec_mcp.server import main

if __name__ == "__main__":
    main()
