LTMD OpenMM Plugin
==================

Long Timestep Molecular Dynamics (LTMD) allows taking timesteps up to 100x larger than traditional MD.  LTMD is written as a plugin for OpenMM.

Requirements
-------------

To use LTMD, you'll need to install:
* CMake
* OpenMM (revision 3324 from SVN -- other versions try to take over all available OpenCL devices and don't respect commands to do otherwise.)
* Gromacs 4.5 (used by ProtoMol to read TPR files)
* [ProtoMol](http://sourceforge.net/projects/protomol/) (used to read TPR files, set up forcefield and simulation parameters) -- take latest from SVN
* LAPACK
* OpenMP (optional)

Installation
--------------

# Compile and install Gromacs 4.5 -- you do not need to compile it against OpenMM 
# Compile and install OpenMM r3324
## Enable OpenCL and CUDA support
## Make sure that it finds the OpenCL.so file. OpenMM will compile without it, but it won't run.  You may need to toggle the advanced options in CMake to see the appropriate field.
# Compile and install LAPACK
# Compile and install OpenMM
## You will need to specify the paths to LAPACK, OpenMM, and the OpenMM source directory
## Do enable the GPU and CUDA 
## Do not enable kernel validation or profiling
## Install LTMDOpenMM to separate directory from OpenMM -- there can be conflicts in terms of the order in which plugins are loaded if you don't
# Compile ProtoMol
## Build ProtoMol using the CMake files located in protomol/src .
## Set lapack type to "LAPACK"
## You'll need to specify paths for LAPACK, Gromacs, OpenMM, LTMD OpenMM
## Enable BUILD_OPENMM, then BUILD_OPENMM_LTMD (do not enable BUILD_OPENMM_FBM -- LTMD includes it on its own)
## ProtoMol does not install so just make it


Running
--------
# Set the OPENMM_PLUGIN_DIR to the OpenMM and LTMD OpenMM plugin directories separated by a colon: "/path/to/openmm/lib/plugin:/path/to/ltmdopenmm/lib/plugin".  (Order is important)
# Run the provided simulation (in examples) as "ProtoMol sim.conf"

