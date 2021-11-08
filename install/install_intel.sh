#!/bin/bash

set -e

trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "$0: \"${last_command}\" command failed with exit code $?"' ERR

sudo apt-get -y install ocl-icd-dev ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers

vendor=$(lscpu | awk '/Vendor ID/{print $3}')
if [[ "$vendor" == "GenuineIntel" ]]; then
  mkdir -p /tmp/neo
  cd /tmp/neo
  wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-gmmlib_20.1.1_amd64.deb
  wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-igc-core_1.0.3826_amd64.deb
  wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-igc-opencl_1.0.3826_amd64.deb
  wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-opencl_20.16.16582_amd64.deb
  wget https://github.com/intel/compute-runtime/releases/download/20.16.16582/intel-ocloc_20.16.16582_amd64.deb
  sudo dpkg -i *.deb
else
    echo "Not installing NEO OpenCL drivers for non-Intel CPU vendor, which is \"$vendor\"."
fi

if [ -e /etc/lightdm ]; then
  sudo sh -c "echo [SeatDefaults]'\n'autologin-user=$USER'\n'autologin-user-timeout=0>/etc/lightdm/lightdm.conf"
fi
