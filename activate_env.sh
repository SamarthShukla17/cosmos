#!/bin/bash
# Activate Cosmos AI environment

if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
    echo "✅ Cosmos AI environment activated!"
    echo "   Python: $(python --version)"
    echo "   Location: $(pwd)"
else
    echo "⚠️  Already in a virtual environment"
fi
