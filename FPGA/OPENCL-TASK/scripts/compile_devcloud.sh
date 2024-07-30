#/bin/bash

cd ../src
time aoc -v -report -board-package=/opt/intel/oneapi/intel_s10sx_pac -I ../include arithmetic.cl -o arithmetic.aocx
