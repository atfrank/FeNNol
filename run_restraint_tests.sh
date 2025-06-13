#!/bin/bash
# Script to run unit tests for the restraints code

# Install the fixed module
echo "Installing fixed restraints module..."
cd /Users/afrank/Downloads/FeNNol-main
cp src/fennol/md/restraints_fixed.py src/fennol/md/restraints.py

# Run the unit tests
echo "Running unit tests for restraints module..."
python test/test_restraints.py

echo ""
echo "Tests complete!"