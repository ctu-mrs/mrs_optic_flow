# general stuff
sudo apt -y install ocl-icd-dev ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers clinfo

# 8th gen intel
cd /tmp
mkdir -p neo
cd neo
wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-gmmlib_20.1.1_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-igc-core_1.0.3826_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-igc-opencl_1.0.3826_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-opencl_20.16.16582_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-ocloc_20.16.16582_amd64.deb
sudo dpkg -i *.deb
