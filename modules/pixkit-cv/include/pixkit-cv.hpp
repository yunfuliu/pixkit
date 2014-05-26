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

		/*
		*	@brief		Non-maximum suppression
		*	@paper		PAMI2010 https://github.com/rbgirshick/voc-dpm/blob/be212abac26859986a0ae16fdf040aa3f9c64a09/test/nms.m
		*				Felzenszwalb, P.F., Girshick, R.B., McAllester, D.A., Ramanan, D.: Object detection with
		*				discriminatively trained part-based models. PAMI 32(9) (2010)
		*	@abbr		pairwise max (PM).
		*
		*	@param		boxes: Detection bounding boxes
		*	@param		overlap: Overlap threshold for suppression. For a selected box Bi, all boxes Bj that are covered by
		*				more than overlap are suppressed. Note that 'covered' is |Bi \cap Bj| / |Bj|, not the PASCAL intersection over
		*				union measure.
		*/
		bool	nms_PairwiseMax_2010(std::vector<CBBs> &boxes,float overlap=0.5);

		/*
		*	@brief		Non-maximum suppression
		*	@paper		P. Dollar, Z. Tu, P. Perona, and S. Belongie, "Integral Channel Features," Proc. British Machine Vision Conf., 2009.
		*	@abbr		pairwise max star (PM*).
		*
		*	@param		boxes: Detection bounding boxes
		*	@param		overlap: Overlap threshold for suppression. For a selected box Bi, all boxes Bj that are covered by
		*				more than overlap are suppressed. Note that 'covered' is |Bi \cap Bj| / |Bi \cup Bj|, is the PASCAL intersection over
		*				union measure.
		*/
		bool	nms_PairwiseMaxStar_2009(std::vector<CBBs> &boxes,float overlap=0.6);

	}

	namespace labeling{

		// Two-pass connected-component labeling
		bool twoPass(const cv::Mat &src,cv::Mat &dst,const int offset);

	}

}
#endif