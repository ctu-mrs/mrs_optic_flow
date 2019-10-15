#!/bin/bash
sudo apt install ocl-icd-dev ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers

cd neo
wget https://github.com/intel/compute-runtime/releases/download/19.11.12599/intel-gmmlib_18.4.1_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.11.12599/intel-igc-core_19.11.1622_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.11.12599/intel-igc-opencl_19.11.1622_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.11.12599/intel-opencl_19.11.12599_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/19.11.12599/intel-ocloc_19.11.12599_amd64.deb
sudo dpkg -i *.deb

echo "[SeatDefaults]
autologin-user=mrs
autologin-user-timeout=0" > /etc/lightdm/lightdm.conf
