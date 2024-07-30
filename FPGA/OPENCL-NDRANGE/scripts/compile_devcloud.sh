#/bin/bash

#source /glob/development-tools/versions/fpgasupportstack/d5005/2.0.1/inteldevstack/init_env.sh
#source /glob/development-tools/versions/fpgasupportstack/d5005/2.0.1/inteldevstack/hld/init_opencl.sh
#export FPGA_BBB_CCI_SRC=/usr/local/intel-fpga-bbb
#export PATH=/glob/intel-python/python2/bin:${PATH}

cd ../src
time aoc -v -report -board-package=/opt/intel/oneapi/intel_s10sx_pac -I ../include arithmetic.cl -o arithmetic.aocx
