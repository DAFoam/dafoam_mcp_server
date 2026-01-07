#!/bin/bash

echo "Cleaning..."
rm -rf postProcessing
rm -rf *.bin *.info *.stl *.jpeg *.png reports *.txt *.hst
rm -rf processor* 0.00* 
rm -rf .dafoam_run_finished
find plots -type f ! -name '*mesh*' -delete
rm -rf {1..9}*

