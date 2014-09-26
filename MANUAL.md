PIXKIT MANUAL
=============

* Install [OpenCV](http://opencv.org/)

* Go [releases](http://goo.gl/GHfv9g "pixkit/releases") and download the latest stable version to for instance, `C:\pixkit\src\`  
  *You can also click the "[Download ZIP](https://github.com/yunfuliu/pixkit)" on the homepage for up-to-date functions. However, the libs in `C:\pixkit\src\lib\` may not support the newer functions which are released after the latest release version.*

	pixkit involves following major functions:
	- computer vision: `pixkit-cv`
	- file process: `pixkit-file`
	- image processing: `pixkit-image`
	- maching learning: `pixkit-ml`
	- timer: `pixkit-timer`

* Function definitions: Please go [wiki](https://github.com/yunfuliu/pixkit/wiki)

* (Optional) If you meet bugs during above procedure or while you are using pixkit, please go [issues](https://github.com/yunfuliu/pixkit/issues) and report in either English or Chinese.

* There are two ways for using pixkit:
  1. Without CMake:
     - 1) Check whether the versions of libs in `C:\pixkit\src\lib` support your environment. If not, you have to use CMake for using pixkit; if yes, go to the next step.
     - 2) For each of your project, involves the following link to your lib path. For instance, the environment with vc10 and x64, you should involve this to your lib path: `C:\pixkit\src\lib\x64\vc10`.
     - 3) For each of your project, involves this link to your source code path: `C:\pixkit\src\modules\`
  2. With CMake:

     - 1) Build up pixkit with [CMake](http://www.cmake.org/) for your environment, and assign a path for your build with, i.e., `C:\pixkit\build\`
	
		By now, only the following environments are tested:
	- Visual Studio 12 Win64 (please select the Visual Studio 11 while selecting your platform)
	- Visual Studio 11 Win32/Win64
	- Visual Studio 10 Win32/Win64

    ![aa](http://miupix.cc/dm/M45HRY/sample.jpg)

     - 2) Compile INSTALL project in `C:\pixkit\build\pixkit.sln` with both release and debug modes

     - 3) All the required resources involving "include" and "lib" are in the folder `C:\pixkit\build\install\`

	- Samples: Some few examples are also built up in `C:\pixkit\build\install\samples\` for test

