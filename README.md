# The Optic flow processing for GPU-denied environments

## System requirements
Either a GPU (Integrated is often sufficient) or a decent CPU.

## Troubleshooting
### GPU accelerated processing
The acceleration is implemented in OpenCL framework. Regardless of your GPU, install packages:
```
ocl-icd-dev ocl-icd-lbopencl1 ocl-icd-opencl-dev opencl-headers
```
and I also recommend to install
```
clinfo
```
for testing if your GPU even shows up as an OpenCL device.

#### Intel GPUs
For generations greater than 8th (Broadwell) you have to install the NEO OpenCL drivers:
https://github.com/intel/compute-runtime/releases

For older (3rd to 6th gen) you have to install the Beignet OpenCL drivers:
https://github.com/intel/beignet

For generations in between, either may or may not work, try it starting with NEO drivers.

The lightdm has to be started for the GPU to show as a OpenCL device. For this to be the case, include in /etc/lightdm/ a file `lightdm.conf`, with the connents:

```
[SeatDefaults]
autologin-user=mrs
autologin-user-timeout=0
```

#### Nvidia GPUs
Processing a single piece of data on an external GPU is faster, but the transfer of data into and from these devices creates a significant overhead, which typically leads to the integrated GPU providing a better output rate.
In integrated GPUs the data is simply being copied to a different area in the same RAM, whereas in a discrete GPU it is copied over the PCI into the GDDR.
This is even worse if the GPU is connected via Thunderbolt.
With new devices, don't assume that one GPU works better, but rather test this empirically.

Also, install packages:
```
nvidia-libopencl1-XXX [nvidia-libopencl1-XXX-updates] nvidia-opencl-icd-XXX nvidia-opencl-dev
```
where XXX is your Nvidia driver version.
In some cases, you may also have to install CUDA Toolkit.


### CPU
If you can't get any GPU to cooperate, enable the CPU processing by passing the parameter useOCL in launcher as false.
This will work everywhere, but the output rate may be less than satisfactory.
Set flight control to appropriate speeds, as the lower rate leads to sticter limits on the maximum measurable speeds - the shift during a longer period may move the ground image out of matching range.
