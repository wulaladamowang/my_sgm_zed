# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wang/code/c++Code/my_sgm_zed

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wang/code/c++Code/my_sgm_zed/build

# Include any dependencies generated for this target.
include CMakeFiles/my_sgm_zed.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/my_sgm_zed.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/my_sgm_zed.dir/flags.make

CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o: CMakeFiles/my_sgm_zed.dir/flags.make
CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o: ../src/get_point_cloud.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o -c /home/wang/code/c++Code/my_sgm_zed/src/get_point_cloud.cpp

CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed/src/get_point_cloud.cpp > CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.i

CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed/src/get_point_cloud.cpp -o CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.s

CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.requires:

.PHONY : CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.requires

CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.provides: CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.requires
	$(MAKE) -f CMakeFiles/my_sgm_zed.dir/build.make CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.provides.build
.PHONY : CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.provides

CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.provides.build: CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o


CMakeFiles/my_sgm_zed.dir/src/little_tips.o: CMakeFiles/my_sgm_zed.dir/flags.make
CMakeFiles/my_sgm_zed.dir/src/little_tips.o: ../src/little_tips.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/my_sgm_zed.dir/src/little_tips.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/little_tips.o -c /home/wang/code/c++Code/my_sgm_zed/src/little_tips.cpp

CMakeFiles/my_sgm_zed.dir/src/little_tips.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/little_tips.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed/src/little_tips.cpp > CMakeFiles/my_sgm_zed.dir/src/little_tips.i

CMakeFiles/my_sgm_zed.dir/src/little_tips.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/little_tips.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed/src/little_tips.cpp -o CMakeFiles/my_sgm_zed.dir/src/little_tips.s

CMakeFiles/my_sgm_zed.dir/src/little_tips.o.requires:

.PHONY : CMakeFiles/my_sgm_zed.dir/src/little_tips.o.requires

CMakeFiles/my_sgm_zed.dir/src/little_tips.o.provides: CMakeFiles/my_sgm_zed.dir/src/little_tips.o.requires
	$(MAKE) -f CMakeFiles/my_sgm_zed.dir/build.make CMakeFiles/my_sgm_zed.dir/src/little_tips.o.provides.build
.PHONY : CMakeFiles/my_sgm_zed.dir/src/little_tips.o.provides

CMakeFiles/my_sgm_zed.dir/src/little_tips.o.provides.build: CMakeFiles/my_sgm_zed.dir/src/little_tips.o


CMakeFiles/my_sgm_zed.dir/src/main.o: CMakeFiles/my_sgm_zed.dir/flags.make
CMakeFiles/my_sgm_zed.dir/src/main.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/my_sgm_zed.dir/src/main.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/main.o -c /home/wang/code/c++Code/my_sgm_zed/src/main.cpp

CMakeFiles/my_sgm_zed.dir/src/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/main.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed/src/main.cpp > CMakeFiles/my_sgm_zed.dir/src/main.i

CMakeFiles/my_sgm_zed.dir/src/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/main.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed/src/main.cpp -o CMakeFiles/my_sgm_zed.dir/src/main.s

CMakeFiles/my_sgm_zed.dir/src/main.o.requires:

.PHONY : CMakeFiles/my_sgm_zed.dir/src/main.o.requires

CMakeFiles/my_sgm_zed.dir/src/main.o.provides: CMakeFiles/my_sgm_zed.dir/src/main.o.requires
	$(MAKE) -f CMakeFiles/my_sgm_zed.dir/build.make CMakeFiles/my_sgm_zed.dir/src/main.o.provides.build
.PHONY : CMakeFiles/my_sgm_zed.dir/src/main.o.provides

CMakeFiles/my_sgm_zed.dir/src/main.o.provides.build: CMakeFiles/my_sgm_zed.dir/src/main.o


CMakeFiles/my_sgm_zed.dir/src/my_camera.o: CMakeFiles/my_sgm_zed.dir/flags.make
CMakeFiles/my_sgm_zed.dir/src/my_camera.o: ../src/my_camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/my_sgm_zed.dir/src/my_camera.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/my_camera.o -c /home/wang/code/c++Code/my_sgm_zed/src/my_camera.cpp

CMakeFiles/my_sgm_zed.dir/src/my_camera.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/my_camera.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed/src/my_camera.cpp > CMakeFiles/my_sgm_zed.dir/src/my_camera.i

CMakeFiles/my_sgm_zed.dir/src/my_camera.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/my_camera.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed/src/my_camera.cpp -o CMakeFiles/my_sgm_zed.dir/src/my_camera.s

CMakeFiles/my_sgm_zed.dir/src/my_camera.o.requires:

.PHONY : CMakeFiles/my_sgm_zed.dir/src/my_camera.o.requires

CMakeFiles/my_sgm_zed.dir/src/my_camera.o.provides: CMakeFiles/my_sgm_zed.dir/src/my_camera.o.requires
	$(MAKE) -f CMakeFiles/my_sgm_zed.dir/build.make CMakeFiles/my_sgm_zed.dir/src/my_camera.o.provides.build
.PHONY : CMakeFiles/my_sgm_zed.dir/src/my_camera.o.provides

CMakeFiles/my_sgm_zed.dir/src/my_camera.o.provides.build: CMakeFiles/my_sgm_zed.dir/src/my_camera.o


# Object files for target my_sgm_zed
my_sgm_zed_OBJECTS = \
"CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o" \
"CMakeFiles/my_sgm_zed.dir/src/little_tips.o" \
"CMakeFiles/my_sgm_zed.dir/src/main.o" \
"CMakeFiles/my_sgm_zed.dir/src/my_camera.o"

# External object files for target my_sgm_zed
my_sgm_zed_EXTERNAL_OBJECTS =

my_sgm_zed: CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o
my_sgm_zed: CMakeFiles/my_sgm_zed.dir/src/little_tips.o
my_sgm_zed: CMakeFiles/my_sgm_zed.dir/src/main.o
my_sgm_zed: CMakeFiles/my_sgm_zed.dir/src/my_camera.o
my_sgm_zed: CMakeFiles/my_sgm_zed.dir/build.make
my_sgm_zed: /usr/local/zed/lib/libsl_zed.so
my_sgm_zed: /usr/lib/x86_64-linux-gnu/libopenblas.so
my_sgm_zed: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
my_sgm_zed: /usr/lib/x86_64-linux-gnu/libcuda.so
my_sgm_zed: /usr/local/cuda-10.0/lib64/libcudart.so
my_sgm_zed: /usr/local/lib/libopencv_gapi.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_stitching.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_aruco.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_bgsegm.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_bioinspired.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_ccalib.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudabgsegm.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudafeatures2d.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudaobjdetect.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudastereo.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_dnn_objdetect.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_dnn_superres.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_dpm.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_face.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_freetype.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_fuzzy.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_hfs.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_img_hash.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_line_descriptor.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_quality.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_reg.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_rgbd.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_saliency.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_stereo.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_structured_light.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_superres.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_surface_matching.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_tracking.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_videostab.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_xfeatures2d.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_xobjdetect.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_xphoto.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_shape.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_highgui.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_datasets.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_plot.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_text.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_dnn.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_ml.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_phase_unwrapping.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudacodec.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_videoio.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudaoptflow.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudalegacy.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudawarping.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_optflow.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_ximgproc.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_video.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_objdetect.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_calib3d.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_features2d.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_flann.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_photo.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudaimgproc.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudafilters.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_imgproc.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudaarithm.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_core.so.4.2.0
my_sgm_zed: /usr/local/lib/libopencv_cudev.so.4.2.0
my_sgm_zed: CMakeFiles/my_sgm_zed.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wang/code/c++Code/my_sgm_zed/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable my_sgm_zed"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_sgm_zed.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/my_sgm_zed.dir/build: my_sgm_zed

.PHONY : CMakeFiles/my_sgm_zed.dir/build

CMakeFiles/my_sgm_zed.dir/requires: CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.requires
CMakeFiles/my_sgm_zed.dir/requires: CMakeFiles/my_sgm_zed.dir/src/little_tips.o.requires
CMakeFiles/my_sgm_zed.dir/requires: CMakeFiles/my_sgm_zed.dir/src/main.o.requires
CMakeFiles/my_sgm_zed.dir/requires: CMakeFiles/my_sgm_zed.dir/src/my_camera.o.requires

.PHONY : CMakeFiles/my_sgm_zed.dir/requires

CMakeFiles/my_sgm_zed.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/my_sgm_zed.dir/cmake_clean.cmake
.PHONY : CMakeFiles/my_sgm_zed.dir/clean

CMakeFiles/my_sgm_zed.dir/depend:
	cd /home/wang/code/c++Code/my_sgm_zed/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wang/code/c++Code/my_sgm_zed /home/wang/code/c++Code/my_sgm_zed /home/wang/code/c++Code/my_sgm_zed/build /home/wang/code/c++Code/my_sgm_zed/build /home/wang/code/c++Code/my_sgm_zed/build/CMakeFiles/my_sgm_zed.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/my_sgm_zed.dir/depend
