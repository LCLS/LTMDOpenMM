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

1. Compile and install Gromacs 4.5 -- you do not need to compile it against OpenMM 
2. Compile and install OpenMM r3324
   1. Enable OpenCL and CUDA support
   2.Make sure that it finds the OpenCL.so file. OpenMM will compile without it, but it won't run.  You may need to toggle the advanced options in CMake to see the appropriate field.
3. Compile and install LAPACK
4. Compile and install OpenMM
   1. You will need to specify the paths to LAPACK, OpenMM, and the OpenMM source directory
   2. Do enable the GPU and CUDA 
   3. Do not enable kernel validation or profiling
   4. Install LTMDOpenMM to separate directory from OpenMM -- there can be conflicts in terms of the order in which plugins are loaded if you don't
5. Compile ProtoMol
   1. Build ProtoMol using the CMake files located in protomol/src .
   2. Set lapack type to "LAPACK"
   3. You'll need to specify paths for LAPACK, Gromacs, OpenMM, LTMD OpenMM
   4. Enable BUILD_OPENMM, then BUILD_OPENMM_LTMD (do not enable BUILD_OPENMM_FBM -- LTMD includes it on its own)
   5. ProtoMol does not install so just make it


Running
--------

1. Set the OPENMM_PLUGIN_DIR to the OpenMM and LTMD OpenMM plugin directories separated by a colon: "/path/to/openmm/lib/plugin:/path/to/ltmdopenmm/lib/plugin".  (Order is important)
2. Run the provided simulation (in examples) as "ProtoMol sim.conf"

