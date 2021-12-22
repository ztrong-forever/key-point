#coding=utf-8
'''
json标签生成txt文件
'''
import os
import json
import cv2


def crop_img(img_path,file):
    img_path=os.path.join(img_path,file)
    img=cv2.imread(img_path)
    mask_file=file.replace('.jpg','_mask.jpg')
    mask_path=os.path.join(mask_dir,mask_file)
    mask=cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    a,b=cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in a]
    for bbox in bounding_boxes:
        [x, y, w, h] = bbox
        aera=w*h
        if aera>=1000 :
            img_crop=img[y-20:y+h+20,x-20:x+w+20]
            left=x-20
            top=y-20
            right=x+w+20
            bottom=y+h+20
    return left,top,right,bottom,img_crop


dir="/home/projectCodes/heart_point/data/json/"
img_dir="/home/projectCodes/heart_point/data/image/"
mask_dir="/home/projectCodes/dataprocess/masks/"
test="/home/projectCodes/heart_point/data/test/"
img_crop_path="/home/projectCodes/heart_point/data/crop"
sets=["train","test","val"]
k=0
for set in sets:
    count=0
    dir_path=os.path.join(dir,set)
    json_path=os.listdir(dir_path)
    # with open('/home/projectCodes/heart_point/'+set+".txt",'w',encoding='utf-8') as f:
    with open('/home/projectCodes/heart_point/'+set+"_crop.txt",'w',encoding='utf-8') as f:#剪裁
        for file in json_path:
            filepath=os.path.join(dir_path,file)
            filename=file[:-12]+".jpg"
            img_path=os.path.join(img_dir,set)
            left,top,right,bottom,img_crop=crop_img(img_path=img_path,file=filename)#剪裁

            k+=1
            f.write(str(os.path.join(img_crop_path,set,file[:-12])+".jpg"))
            with open(filepath,'r',encoding='utf8')as fp:
                json_data=json.load(fp)
                for data in json_data["shapes"]:
                    label=data["label"]
                    point=data["points"][0]
                    point_x=round(point[0]-left,4)
                    point_y=round(point[1]-top,4)
                    cv2.circle(img_crop, (int(point_x),int(point_y)), 1, (0,0,255), 4)

                    f.write(" "+str(label)+" "+str(point_x)+" "+str(point_y))
                cv2.imwrite("/home/projectCodes/heart_point/test/"+str(k)+".jpg",img_crop)
                f.write("\n")    
            count+=1
    print(set+":%d"%count)
    f.close()
            
