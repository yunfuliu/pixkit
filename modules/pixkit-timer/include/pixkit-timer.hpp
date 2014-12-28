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
#include <iostream>
#include <cassert>

namespace pixkit{

	class Timer{
	public:	
		Timer(const std::string t = "Timer"):title(t){
			is_started = false; start_clock = 0; cumulative_clock = 0; n_starts = 0;
		}
		~Timer(){	if (is_started) printf("pixkit::timer '%s' is started and is being destroyed.\n", title.c_str());	}

		void Start();
		void Stop();
		void Reset();

		float Report();
		float StopAndReport() { Stop(); return Report(); }
		float TimeInSeconds();

		float AvgTime(){assert(is_started == false); return TimeInSeconds()/n_starts;}

	private:
		const std::string title;

		bool is_started;
		unsigned long long int start_clock;
		unsigned long long int cumulative_clock;
		unsigned int n_starts;
	};
}