=====================================================
 CESM 2.1.5 — CAMulator Coupled Configuration
=====================================================

This branch (``release-cesm2.1.5-camulator``) extends stock CESM 2.1.5 to support
**CAMulator**: a machine-learning atmosphere that replaces CAM6 in a fully coupled
ocean–ice–land simulation. Atmospheric state is predicted by a neural network
(trained via `CREDIT <https://github.com/NCAR/miles-credit>`_) and coupled to the
active ocean (POP2), sea ice (CICE5), and land (CLM5) components through a
file-based protocol.

.. contents::
   :local:
   :depth: 2

What is different from stock CESM 2.1.5
=========================================

Two components are replaced:

+------------+----------------------------+------------------------------------------------------+
| Component  | Stock CESM 2.1.5           | This branch                                          |
+============+============================+======================================================+
| CIME       | ESMCI/cime @ maint-5.6     | Cambridge-ICCS/cime_je @ coupled_camulator           |
+------------+----------------------------+------------------------------------------------------+
| CAM        | ESCOMP/CAM @ cam_rel60     | WillyChap/CAM @ coupled_camulator                    |
+------------+----------------------------+------------------------------------------------------+
| All others | Stock CESM 2.1.5 tags      | Unchanged                                            |
+------------+----------------------------+------------------------------------------------------+

Key additions in CIME:

* ``datm_datamode_camulator.F90`` — file-based coupling protocol between CESM and the CAMulator Python server
* ``CAMULATOR`` registered as a valid DATM datamode
* FTorch build infrastructure (``USE_FTORCH`` flag)
* Fix for SST=0 bug when using DATA atmosphere with an active ocean

Requirements
============

System
------

* NCAR Derecho (tested) — other machines require porting ``config_machines.xml``
* At least 1× A100 GPU for the CAMulator Python server

Python environment (CAMulator server)
--------------------------------------

Install the CREDIT conda environment from the
`CREDIT repository <https://github.com/NCAR/miles-credit>`_::

    conda activate credit-coupling
    pip install -e . --no-deps

Pre-trained model checkpoint::

    /glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/CAMulator_models/checkpoint.pt00091.pt

Contact wchapman@ucar.edu for access or details.

Installation
============

::

    git clone -b release-cesm2.1.5-camulator \
        https://github.com/WillyChap/CESM.git my_camulator_cesm
    cd my_camulator_cesm
    ./manage_externals/checkout_externals

Creating and building a case
=============================

::

    cd cime/scripts

    ./create_newcase \
        --case /glade/work/$USER/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v01 \
        --compset GIAF \
        --res f09_g17 \
        --mach derecho \
        --project <YOUR_PROJECT>

    cd /glade/work/$USER/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v01

    # Switch atmosphere to CAMulator data mode
    ./xmlchange DATM_MODE=CAMULATOR

    # Required MPI/GPU environment fixes on Derecho
    ./xmlchange --file env_mach_specific.xml MPICH_GPU_SUPPORT_ENABLED=0
    ./xmlchange --file env_mach_specific.xml FI_CXI_DISABLE_HOST_REGISTER=1
    ./xmlchange --file env_mach_specific.xml MPICH_SMP_SINGLE_COPY_MODE=NONE

    ./case.setup
    ./case.build

Running
=======

CAMulator requires **two processes running simultaneously**.

1. Start the CAMulator Python server (GPU node)
-------------------------------------------------

On a Casper GPU node::

    # qsub -I -A <PROJECT> -l select=1:ncpus=32:ngpus=1:mem=250GB \
    #      -l walltime=12:00:00 -q casper -l gpu_type=a100_80gb

    conda activate credit-coupling
    cd /path/to/miles-credit/climate

    python camulator_server.py \
        --config ./camulator_config.yml \
        --model_name checkpoint.pt00091.pt \
        --rundir /glade/derecho/scratch/$USER/g.e21.CAMULATOR_GIAF_v01/run/ \
        --save_atm_nc camulator_out \
        --daily_mean

The server waits for CESM to write ``camulator_go.flag``, runs one inference step,
writes ``camulator_cam_out.nc``, then signals CESM via ``camulator_done.flag``.

The server **must be running before CESM reaches its first coupling step**.

2. Submit the CESM job
-----------------------

::

    cd /glade/work/$USER/cesm/CREDIT/g.e21.CAMULATOR_GIAF_v01
    ./case.submit

Key configuration
==================

Edit ``climate/camulator_config.yml`` in the CREDIT repo before each run:

.. code-block:: yaml

    predict:
      save_forecast: '/glade/derecho/scratch/<USER>/CREDIT/climate_output/'
      init_cond_fast_climate: '/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/init_condition_tensor_2000-01-01T00Z.pth'
      start_datetime: '2000-01-01 00:00:00'
      timesteps_fast_climate: 1460   # 6-hr steps = 1 year

Output
======

* **Coupled CESM output**: standard history files (ocean, ice, land) in the run directory
* **Atmospheric output**: ``<rundir>/camulator_out/YYYY/camulator.h1.<YYYY-MM-DD-SSSSS>.nc``
* **Daily means**: ``<rundir>/camulator_out/camulator.h1d.<YYYY-MM-DD>.nc``

.. note::
   Move output from ``/scratch/`` to ``/campaign/`` storage — scratch has a 90-day purge policy.

Restarting
==========

The server saves an atmosphere restart file after every step::

    <rundir>/camulator_atm_restart.pth

On restart, CESM resumes normally and the server automatically loads this file.
To start a **fresh run** from the same case, delete this file before relaunching the server.
Annual restart archives are saved to ``<rundir>/atm_restarts/``.

Known limitations
=================

* Tested on NCAR Derecho/Casper only. Other machines require entries in
  ``cime/config/cesm/machines/config_machines.xml`` and ``config_compilers.xml``.
* The neural network predicts FSNS and FLNS; downwelling fluxes (FSDS, FLNSD) are
  reconstructed from these. Direct prediction is planned for the next training cycle.
* Performance: ~45 simulated years per wall-clock day (SYPD) on Derecho + 1× A100.

Citation
=========

If you use this configuration please cite::

    Chapman et al. (2025), CAMulator: A Machine Learning Emulator of CAM6
    for Long-Running Coupled Climate Simulations [citation TBD]

and the standard CESM 2.1.5 references.

Contact
=======

Will Chapman — wchapman@ucar.edu — MILES Group, NSF NCAR

----

Stock CESM 2.1.5 documentation
================================

The remainder of this file is the original CESM README, retained for reference.

==================================
 The Community Earth System Model
==================================

See the CESM web site for documentation and information:

http://www.cesm.ucar.edu

The CESM Quickstart Guide is available at:

http://escomp.github.io/cesm

This repository provides tools for managing the external components that
make up a CESM tag - alpha, beta and release. CESM tag creation should
be coordinated through CSEG at NCAR.

.. sectnum::

Software requirements
=====================

Software requirements for installing, building and running CESM
---------------------------------------------------------------

Installing, building and running CESM requires:

* a Unix-like operating system (Linux, AIX, OS X, etc.)

* git client version 1.8 or newer

* subversion client (we have tested with versions 1.6.11 and newer)

* python2 version 2.7 or newer (cime supports python3, but some CESM components are not python3-compliant)

* perl version 5

* build tools gmake and cmake

* Fortran and C compilers

  * See `Details on Fortran compiler versions`_ below for more information

* LAPACK and BLAS libraries

* a NetCDF library version 4.3 or newer built with the same compiler you
  will use for CESM

  * a PnetCDF library is optional

* a functioning MPI environment (unless you plan to run on a single core
  with the CIME mpi-serial library)

Details on Fortran compiler versions
------------------------------------
The Fortran compiler must support Fortran 2003 features. However, even
among mainstream Fortran compilers that claim to support Fortran 2003,
we have found numerous bugs. Thus, many compiler versions do *not* build
or run CESM properly (see
https://wiki.ucar.edu/display/ccsm/Fortran+Compiler+Bug+List for more
details on older Fortran compiler versions).

CESM2 is tested on several different systems with newer Fortran compilers:
Please see `CESM2.0 Compiler/Machine Tests <https://docs.google.com/spreadsheets/d/15QUqsXD1Z0K_rYNTlykBvjTRt8s0XcQw0cfAj9DZbj0/edit#gid=0>`_
for a spreadsheet of the current results.

More details on porting CESM
----------------------------

For more details on porting CESM to a new machine, see
http://esmci.github.io/cime/users_guide/porting-cime.html

Obtaining the full model code and associated scripting infrastructure
=====================================================================

CESM2.0 is now released via github. You will need some familiarity with git in order
to modify the code and commit these changes. However, to simply checkout and run the
code, no git knowledge is required other than what is documented in the following steps.

To obtain the CESM2.0 code you need to do the following:

#. Clone the repository. ::

      git clone https://github.com/escomp/cesm.git my_cesm_sandbox

   This will create a directory ``my_cesm_sandbox/`` in your current working directory.

#. Run the script **manage_externals/checkout_externals**. ::

      ./manage_externals/checkout_externals

   The **checkout_externals** script is a package manager that will
   populate the cesm directory with the relevant versions of each of the
   components along with the CIME infrastructure code.

At this point you have a working version of CESM.

To see full details of how to set up a case, compile and run, see the CIME documentation at http://esmci.github.io/cime/ .
