def find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block):
    draw_img = np.copy(img)
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
            subimg = cv2.resize(img_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            test_features = train.combine_feat(hog_feat, subimg)
            test_features = test_features.reshape(1,-1)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img
    
ystart = 400
ystop = 656
scale = 1.5
orient = 9
pix_per_cell = 8
cell_per_block = 2
    
out_img = find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block)

fig = plt.figure(figsize=(20,8))
plt.imshow(out_img)