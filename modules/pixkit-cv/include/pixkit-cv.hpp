//////////////////////////////////////////////////////////////////////////
// 
// SOURCE CODE: https://github.com/yunfuliu/pixkit
// 
// BEIRF: pixkit-cv contains computer vision related tools and methods which 
//		  have been published (on articles, e.g., journal or conference papers).
// 
//////////////////////////////////////////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cmath>
#include <vector>


#ifndef __PIXKIT_CV_HPP__
#define __PIXKIT_CV_HPP__

namespace pixkit{

	namespace detection{

		// bounding box
		class CBBs{
		public:
			cv::Rect	rect;	// position of that BB
			float		conf;	// confidence
		};

		bool	nms_PairwiseMax_2010(std::vector<CBBs> &boxes,float overlap=0.5);

		bool	nms_PairwiseMaxStar_2009(std::vector<CBBs> &boxes,float overlap=0.6);

	}

	namespace labeling{

		// Two-pass connected-component labeling
		bool twoPass(const cv::Mat &src,cv::Mat &dst,const int offset=1);

	}

}
#endif
