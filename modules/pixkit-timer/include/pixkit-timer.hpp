//========================================================================
//
// pixkit-timer.hpp
// repo: https://github.com/yunfuliu/pixkit
//
// Prototype of this comes from the CmTimer.h
// in repo: https://github.com/MingMingCheng/CmCode
//
// Author: Ming-Ming Cheng
// 
//========================================================================

#pragma once
#include <ctime>
#include <iostream>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace pixkit{

	class Timer{
	public:	
		Timer::Timer(const std::string t = "Timer"):title(t){
			is_started = false; start_clock = 0; cumulative_clock = 0; n_starts = 0;
		}
		~Timer(){	if (is_started) printf("pixkit::timer '%s' is started and is being destroyed.\n", title.c_str());	}

		inline void Start();
		inline void Stop();
		inline void Reset();

		inline bool Report();
		inline bool StopAndReport() { Stop(); return Report(); }
		inline float TimeInSeconds();

		inline float AvgTime(){assert(is_started == false); return TimeInSeconds()/n_starts;}

	private:
		const std::string title;

		bool is_started;
		unsigned int start_clock;
		unsigned int cumulative_clock;
		unsigned int n_starts;
	};
}

inline void pixkit::Timer::Start(){
	if (is_started){
		printf("pixkit::timer '%s' is already started. Nothing done.\n", title.c_str());
		start_clock = cvGetTickCount();
		return;
	}
	is_started = true;	n_starts++;	start_clock = cvGetTickCount();
}
inline void pixkit::Timer::Stop(){
	if (!is_started){
		printf("pixkit::timer '%s' is started. Nothing done\n", title.c_str());
		return;
	}
	cumulative_clock += cvGetTickCount() - start_clock;	is_started = false;
}
inline void pixkit::Timer::Reset(){
	if (is_started)	{
		printf("pixkit::timer '%s'is started during reset request.\n Only reset cumulative time.\n");
		return;
	}
	start_clock = 0;	cumulative_clock = 0;	n_starts = 0;
}
inline bool pixkit::Timer::Report(){
	if (is_started){
		printf("pixkit::timer '%s' is started.\n Cannot provide a time report.", title.c_str());
		return false;
	}
	float timeUsed = TimeInSeconds();
	printf("[%s] CumuTime: %4gs, #run: %4d, AvgTime: %4gs\n", title.c_str(), timeUsed, n_starts, timeUsed/n_starts);
	return true;
}
inline float pixkit::Timer::TimeInSeconds(){
	if (is_started){
		printf("pixkit::timer '%s' is started. Nothing done\n", title.c_str());
		return 0;
	}
	return float(cumulative_clock) / cv::getTickFrequency();
}