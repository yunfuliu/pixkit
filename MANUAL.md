PIXKIT MANUAL
=============

* Install [OpenCV](http://opencv.org/)

* Go [releases](http://goo.gl/GHfv9g "pixkit/releases") and download the latest stable version to for instance, `C:\pixkit\src\`  
  *You can also click the "[Download ZIP](https://github.com/yunfuliu/pixkit)" on the homepage for up-to-date functions.*

	pixkit involves following major functions:
	- image processing: `pixkit-image`
	- maching learning: `pixkit-ml`
	- timer: `pixkit-timer`

* Build up pixkit with [CMake](http://www.cmake.org/) for your environment, and assign a path for your build with, i.e., `C:\pixkit\build\`
	
	By now, only the following environments are tested:
	- Visual Studio 12 Win64 (please select the Visual Studio 11 while selecting your platform)
	- Visual Studio 11 Win32/Win64
	- Visual Studio 10 Win32/Win64

    ![aa](http://miupix.cc/dm/M45HRY/sample.jpg)

* Compile INSTALL project in `C:\pixkit\build\pixkit.sln` with both release and debug modes

* All the required resources involving "include" and "lib" are in the folder `C:\pixkit\build\install\`

	- Samples: Some few examples are also built up in `C:\pixkit\build\install\samples\` for test

	- Function definitions: Please go [wiki](https://github.com/yunfuliu/pixkit/wiki)

* (Optional) If you meet bugs during above procedure or while you are using pixkit, please go [issues](https://github.com/yunfuliu/pixkit/issues) and report in either English or Chinese.
