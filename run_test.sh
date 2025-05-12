#!/bin/bash
# Script to install and test the fixed restraints code

# Install the updated code
echo "Installing FeNNol with fixes..."
pip install -e .

# Run the regular restraint test
echo "Running the regular restraint test..."
cd examples/md/rna
python -m fennol.md.dynamic input_debug.fnl

# Wait a moment
echo ""
echo "-------------------------------------"
echo "Now testing flat-bottom restraints..."
echo "-------------------------------------"
echo ""
sleep 2

# Run the flat-bottom restraint test
python -m fennol.md.dynamic input_flatbottom.fnl