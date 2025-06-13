#!/bin/bash
# Script to debug restraint behavior

cd /Users/afrank/Downloads/FeNNol-main
echo "Running minimal restraint test (should work even with memory constraints)..."
cd examples/md/rna

# Run the minimal test first (should always work)
python -m fennol.md.dynamic minimal_restraint_test.fnl

# If the simulation produces a colvars file, analyze it
if [ -f "minimal_test.colvars" ]; then
  echo "Analyzing minimal test colvars data..."
  ./analyze_restraints.py minimal_test.colvars
else
  echo "No colvars file found for minimal test."
fi

echo ""
echo "If minimal test works, try the larger system test..."
echo "You can run: python -m fennol.md.dynamic input_restraint_debug.fnl"
echo "Then analyze: ./analyze_restraints.py sub_system_premodified_epoxide_minimized_solv.colvars"