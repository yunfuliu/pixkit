//========================================================================
//
// pixkit-timer.cpp
// repo: https://github.com/yunfuliu/pixkit
//
// Prototype of this comes from the CmTimer.h
// in repo: https://github.com/MingMingCheng/CmCode
//
// Author: Ming-Ming Cheng
// 
//========================================================================

#include "../include/pixkit-timer.hpp"
#include <opencv2/highgui/highgui.hpp>

void pixkit::Timer::Start(){
	if (is_started){
		printf("pixkit::timer '%s' is already started. Nothing done.\n", title.c_str());
		start_clock = cvGetTickCount();
		return;
	}
	is_started = true;	n_starts++;	start_clock = cvGetTickCount();
}
void pixkit::Timer::Stop(){
	if (!is_started){
		printf("pixkit::timer '%s' is started. Nothing done\n", title.c_str());
		return;
	}
	cumulative_clock += cvGetTickCount() - start_clock;	is_started = false;
}
void pixkit::Timer::Reset(){
	if (is_started)	{
		printf("pixkit::timer '%s'is started during reset request.\n Only reset cumulative time.\n");
		return;
	}
	start_clock = 0;	cumulative_clock = 0;	n_starts = 0;
}
float pixkit::Timer::Report(){
	if (is_started){
		printf("pixkit::timer '%s' is started.\n Cannot provide a time report.", title.c_str());
		return false;
	}
	float timeUsed = TimeInSeconds();
	printf("[%s] CumuTime: %4gs, #run: %4d, AvgTime: %4gs\n", title.c_str(), timeUsed, n_starts, timeUsed/n_starts);
	return timeUsed/(float)n_starts;
}
float pixkit::Timer::TimeInSeconds(){
	if (is_started){
		printf("pixkit::timer '%s' is started. Nothing done\n", title.c_str());
		return 0;
	}
	return static_cast<float>(double(cumulative_clock) / cv::getTickFrequency());
}