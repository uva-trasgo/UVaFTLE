
for file in ftle arithmetic preprocess; do

icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70  -O3 -DONEAPI -DBLOCK=512 -DCUDA_DEVICE -I./include/ -c src/$file.cpp -o $file.o -DCUDA_DEVICE 

done 

icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 -O3 -DONEAPI -DBLOCK=512 -DCUDA_DEVICE -I./include/ arithmetic.o  ftle.o preprocess.o -o ftle-oneApi-usm-dev


