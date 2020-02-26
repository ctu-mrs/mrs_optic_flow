^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package mrs_optic_flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.0.5 (2020-02-26)
------------------
* nothing changed here

0.0.4 (2020-02-18)
------------------
* uncommented install ocl
* added parametrized node_start_delay
* added version check
* added sudo to install script
* longrange off for uav and simulation
* Pushing the compensation to oputput again
* Fixing overcompensation
* Adding output of tilt correction effect
* exponential filter off, correction plus
* Fixing cam yaw
* Correcting dynamic tilts for the long range mode
* Changing height for long range estimate to tilted
* Setting unknown outputted values in long-range mode to nans
* Setting long range messages to fcu
* Adding a config-based selection of when to use the long range mode
* Adding condition of using long range mode in Landoff tracker
* Fixes. Also, setting the long range ratio to 4 by default
* Implemented long range mode. Still buggy and will need testing on NUC due to flakyness of my GPU's memory
* optflow twist in body frame
* Updating the launcher for the new tf naming convention
* Contributors: Matej Petrlik, Matej Petrlik (desktop), Tomas Baca, Viktor Walter, delta, foxtrot

0.0.3 (2019-10-25)
------------------
* updated camera frame
* updated launch file, removed old configs and launch files
* Fixing a duplicate attribute error
* adding an install script
* updated optflow camera name
* Created a check of transform being received
* Update optflow camera frame to match new simulation frames
* rehauling launch files
* set correst camera yaw
* Contributors: Pavel Petráček, Tomas Baca, UAV_44, Viktor Walter, uav43, uav46

0.0.2 (2019-07-01)
------------------
* mild refactoring, removed yaw offset from config
* added delay to optic flow launch files
* Contributors: Tomáš Báča, Viktor Walter,

0.0.1 (2019-05-20)
------------------
