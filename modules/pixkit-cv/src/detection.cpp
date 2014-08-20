#include "../include/pixkit-cv.hpp"







bool comparator(const pixkit::detection::CBBs &l, const pixkit::detection::CBBs &r){ 
	return l.conf > r.conf;
}
bool pixkit::detection::nms_PairwiseMax_2010(std::vector<CBBs> &boxes,float overlap){
	if(boxes.size()<=0){
		return false;
	}else{
		// sort pick
		std::sort(boxes.begin(),boxes.end(),comparator);
		// erase others
		std::vector<CBBs>::iterator it_major,it_minor;
		int	idx_major	=	0;
		while(idx_major<boxes.size()){
			it_major	=	boxes.begin()+idx_major;	
			// erase
			it_minor			=	it_major;
			int		idx_minor	=	1;
			while(idx_major+idx_minor<boxes.size()){
				it_minor	=	it_major	+	idx_minor;
				// get overlap
				cv::Rect int_rect	=	it_major->rect & it_minor->rect;	// intersection rect
				float	overlap_ratio	=	((float)int_rect.area())/((float)it_minor->rect.area());
				if(overlap_ratio>overlap){	
					boxes.erase(it_minor);	// erase
				}else{
					idx_minor	++;
				}
			}
			idx_major++;
		}
		return true;
	}
	return true;
}
bool pixkit::detection::nms_PairwiseMaxStar_2009(std::vector<CBBs> &boxes,float overlap){
	if(boxes.size()<=0){
		return false;
	}else{
		// sort pick
		std::sort(boxes.begin(),boxes.end(),comparator);
		// erase others
		std::vector<CBBs>::iterator it_major,it_minor;
		int	idx_major	=	0;
		while(idx_major<boxes.size()){
			it_major	=	boxes.begin()+idx_major;	
			// erase
			it_minor			=	it_major;
			int		idx_minor	=	1;
			while(idx_major+idx_minor<boxes.size()){
				it_minor	=	it_major	+	idx_minor;
				// get overlap
				cv::Rect int_rect	=	it_major->rect & it_minor->rect;	// intersection rect
				float	union_area_size	=	it_major->rect.area()	+	it_minor->rect.area()	-	int_rect.area();
				float	overlap_ratio	=	((float)int_rect.area())/union_area_size;
				if(overlap_ratio>overlap){
					boxes.erase(it_minor);	// erase
				}else{
					idx_minor	++;
				}
			}
			idx_major++;
		}
		return true;
	}
	return true;
}













