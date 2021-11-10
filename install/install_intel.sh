#!/bin/bash

set -e

trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "$0: \"${last_command}\" command failed with exit code $?"' ERR

unattended=0
for param in "$@"
do
  echo $param
  if [[ $param == "--unattended" ]]; then
    echo "installing in unattended mode"
    unattended=1
    subinstall_params="--unattended"
  fi
done

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

default=n
while true; do
  if [[ "$unattended" == "1" ]]
  then
    resp=y
  else
    [[ -t 0 ]] && { read -t 10 -n 2 -p $'\e[1;32mEnable autologin in lightdm? Should be enabled only for real UAVs!!! [y/n] (default: '"$default"$')\e[0m\n' resp || resp=$default ; }
  fi
  response=`echo $resp | sed -r 's/(.*)$/\1=/'`

  if [[ $response =~ ^(y|Y)=$ ]]
  then

    if [ -e /etc/lightdm ]; then
      sudo sh -c "echo [SeatDefaults]'\n'autologin-user=$USER'\n'autologin-user-timeout=0>/etc/lightdm/lightdm.conf"
    fi

    break

  elif [[ $response =~ ^(n|N)=$ ]]
  then

    break
  else
    echo " What? \"$resp\" is not a correct answer. Try y+Enter."
  fi
done
