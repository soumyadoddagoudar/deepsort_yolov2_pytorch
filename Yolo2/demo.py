from Yolo2.utils import *
from Yolo2.darknet import Darknet
import cv2
import time

class Yolo2(object):
	def __init__(self, cfgfile, weightfile,is_xywh=False):
		
		self.m = Darknet(cfgfile)
		self.m.print_network()
		self.m.load_weights(weightfile)
    
		print('Loading weights from %s... Done!' % (weightfile))

		if self.m.num_classes == 20:
			namesfile = 'Yolo2/data/voc.names'
		elif self.m.num_classes == 80:
			namesfile = 'Yolo2/data/coco.names'
		else:
			namesfile = 'Yolo2/data/names'
			
		self.class_names = load_class_names(namesfile)
		self.is_xywh = is_xywh
	 
		self.use_cuda = 1
		if self.use_cuda:
			self.m.cuda()
			
	def __call__(self, ori_img):
		sized = cv2.resize(ori_img, (self.m.width, self.m.height))
		boxes = do_detect(self.m, sized, 0.5, 0.4, self.use_cuda)
		print("----------------",boxes)
		#draw_img = plot_boxes_cv2(org_img, bboxes, None, class_names)
		
		width = ori_img.shape[1]
		height = ori_img.shape[0]
		if len(boxes)==0:
			return None,None,None
        
		    
		boxes = np.vstack(boxes)
		bbox = np.empty_like(boxes[:,:4])
		if self.is_xywh:
			# bbox x y w h
			bbox[:,0] = boxes[:,0]*width
			bbox[:,1] = boxes[:,1]*height
			bbox[:,2] = boxes[:,2]*width
			bbox[:,3] = boxes[:,3]*height
		else:
			# bbox xmin ymin xmax ymax
			bbox[:,0] = (boxes[:,0]-boxes[:,2]/2.0)*width
			bbox[:,1] = (boxes[:,1]-boxes[:,3]/2.0)*height
			bbox[:,2] = (boxes[:,0]+boxes[:,2]/2.0)*width
			bbox[:,3] = (boxes[:,1]+boxes[:,3]/2.0)*height
		cls_conf = boxes[:,5]
		cls_ids = boxes[:,6]
		return bbox, cls_conf, cls_ids
	


		
	

