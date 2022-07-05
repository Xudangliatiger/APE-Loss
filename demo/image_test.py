from mmdet.apis import inference_detector, init_detector, show_result_pyplot

#%%

config = '../configs/ape_loss/ape_loss_arps_r50_fpn_giou_sigmoid_8_24e_coco1333_dens.py'
checkpoint = '/mnt/data0/home/dengjinhong/xdl/APE/work_dirs/ape_loss_arps_r50_fpn_giou_sigmoid_8_24e_coco1333_dens_41.5/epoch_24.pth'
model = init_detector(config, checkpoint, device='cuda:0')
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

#%%
res = []
for i in range(10000):

    img = '../COMP5329S1A2Dataset/data/{}.jpg'.format(str(30000+i))
    # show_result_pyplot(model, img, result, score_thr=0.5)
    result = inference_detector(model, img)
    #show_result_pyplot(model, img, result, score_thr=0.5)


    # max_score = 0
    # max_j = 0
    # for j in range(18):
    #     if result[j].any():
    #         if max_score<result[j][:, 4].max():
    #
    #             max_score =result[j][:, 4].max()
    #             max_j = jbuz

    cls = []
    for j in range(18):
        if result[j].any():
            if result[j][:, 4].max() >0.5: # or j==max_j:

                cls.append(j)

    res.append(cls)
#%%
with open('./result.csv','w') as f:
    f.write("ImageID,Labels\n")
    for idx, clss in enumerate(res):

        img_classes = []

        f.write('{img}.jpg,'.format(img = 30000+idx))

        for cls in clss:
            if cls <11:
                f.write(' {}'.format(cls+1))
            elif cls>=11 and cls<=17:
                f.write(' {}'.format(cls+2))

        f.write('\n')



# show_result_pyplot(model, img, result, score_thr=0.5)
