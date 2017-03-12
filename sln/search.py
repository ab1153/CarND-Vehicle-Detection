from sln import *
from scipy.ndimage.measurements import label
from collections import deque

def find_cars(img, svc, ystart, ystop, orient, pix_per_cell, cell_per_block, scale):
    # draw_img = np.copy(img)
    img_tosearch = img[ystart:ystop,:,:]
    if scale != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    YCrCb = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    
    # Define blocks and steps as above
    nxblocks = (img_tosearch.shape[1] // pix_per_cell)-1
    nyblocks = (img_tosearch.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = train.get_hog_features(YCrCb[...,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = train.get_hog_features(YCrCb[...,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = train.get_hog_features(YCrCb[...,2], orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    boxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat = np.concatenate([hog_feat1,hog_feat2,hog_feat3])
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            # Extract the image patch
            test_features = hog_feat.reshape(1,-1)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                box = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                boxes.append(box)
                
    return boxes


def draw_bboxes(img, bboxes):
    out_img = np.copy(img)
    for bbox in bboxes:
        left_top, right_bottom = bbox
        cv2.rectangle(out_img, left_top, right_bottom, (0,0,255),6)

    return out_img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def draw_result(img, bboxes, threshold=1, output_heapmap=False):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)    
    # Add heat to each box in box list
    heat = add_heat(heat,bboxes)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    if output_heapmap:
        return draw_img, heatmap
    else:
        return draw_img

def find_cars_multiscale(img, svc, ystart, ystop, orient, pix_per_cell, cell_per_block, scales=[1.0, 1.5]):
    all_bboxes = []
    for scale in scales:
        bboxes = find_cars(img, svc, ystart, ystop, orient, pix_per_cell, cell_per_block, scale)
        for bbox in bboxes:
            all_bboxes.append(bbox)    
    
    return all_bboxes

def process(img, svc):
    ystart = 400
    ystop = 656
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    bboxes = find_cars_multiscale(img, svc, ystart, ystop, orient, pix_per_cell, cell_per_block)
    out_img = draw_result(img, bboxes, threshold=5)
    return out_img

class Tracker(object):
    def __init__(self, svc):
        self.svc = svc
        self.ystart = 400
        self.ystop = 656
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.last_bboxes = []

    def process(self, img):
        
        new_bboxes = find_cars_multiscale(img, self.svc, 
            self.ystart, self.ystop, self.orient, self.pix_per_cell, self.cell_per_block,
            scales=[1.0])

        bboxes = self.last_bboxes + new_bboxes
        self.last_bboxes = new_bboxes
        
        size_classes = []
        for bbox in bboxes:
            # h = bbox[1][1] - box[0][1]
            w = bbox[1][0] - bbox[0][0]
            if w not in size_classes:
                size_classes.append(w)

        n_class = len(size_classes)
            
        out_img = draw_result(img, bboxes, n_class)

        return out_img