from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2,os

def pad_img(img):
    ret = cv2.copyMakeBorder(img, 0, 36, 0, 36, cv2.BORDER_CONSTANT, value=(0,0,0))
    return ret

def pad_mask(img):
    ret = cv2.copyMakeBorder(img, 0, 36, 0, 36, cv2.BORDER_CONSTANT, value=1)
    return ret

def crop(img,split_size,stride,img_name,save_path,mode,dir_type):
    index = 0
    for y in range(0, img.shape[0], stride):
        for x in range(0, img.shape[1], stride):
            # img_tile_cut = img[y:y + split_size, x:x + split_size,:]
            # mask_tile_cut = mask[y:y + split_size, x:x + split_size]
            img_tile_cut = img[y:y + split_size, x:x + split_size]
            cur_name = img_name + str(index) + ".png"
            # cv2.imwrite(save_path+mode+"img/"+cur_name,img_tile_cut)
            # cv2.imwrite(save_path+mode+"mask/"+cur_name,mask_tile_cut)
            cv2.imwrite(save_path + mode+ "/" + dir_type + "/"+cur_name,img_tile_cut)
            index+=1
    print("total img:",index)


if __name__ == "__main__":

    path = "/root/autodl-tmp/Massachusetts/"
    save_path = "/root/autodl-tmp/Massa_512/"
    modes = ["train", "val", "test"]
    dirs = ["images", "masks"]

    # 遍历所有模式和目录类型，创建相应的保存路径
    for mode in modes:
        for dir_type in dirs:
            if not os.path.exists(save_path + mode + "/" + dir_type + "/"):
                os.makedirs(save_path + mode + "/" + dir_type + "/")
            cnt = 0
            for img_name in os.listdir(path+ mode + "_" + dir_type + "/"):
                if img_name[-1] == "g":
                    pure_name = img_name.split(".")[0]
                    # print(pure_name)
                    
                    img = cv2.imread(path+ mode + "_" + dir_type + "/"+img_name,cv2.IMREAD_UNCHANGED)

                    if dir_type != "masks":
                        img_pad = pad_img(img)
                    else:
                        img_pad = pad_mask(img)

                    # crop(img_pad,mask_pad,512,512,pure_name,save_path,mode)
                    crop(img_pad,512,512,pure_name,save_path,mode,dir_type)

                    cnt+=1
            print(cnt)
    
#     for img_name in os.listdir(path+"test/"):
#         if img_name[-1] == "g":
#             pure_name = img_name.split(".")[0]
#             # print(pure_name)
#             # img = cv2.imread(path+"test_images/"+img_name,cv2.IMREAD_UNCHANGED)
#             # mask = cv2.imread(path+"test_masks/"+img_name,cv2.IMREAD_UNCHANGED)
#             label = cv2.imread(path+"test_labels/"+img_name,cv2.IMREAD_UNCHANGED)

#             # img_pad = pad_img(img)
#             # mask_pad = pad_mask(mask)
#             img_pad = pad_img(label)

#             # crop(img_pad,mask_pad,512,512,pure_name,save_path,mode)
#             crop(img_pad,512,512,pure_name,save_path,mode)

#             #print(mask_pad.shape)
#             cnt+=1
#     print(cnt)
    
    



