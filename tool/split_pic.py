# coding: utf-8
"""
分图片
"""
import os
import shutil
txtpath="/home/projectCodes/heart_point/data" #数据集txt文件路径(.json)
path="/home/projectCodes/heart_point/data"  #输出图片路径
dir="/home/projectCodes/dataprocess" #原路径
folder=["crop"]
sets=["train","test","val"]

for setname in sets:
    for foldername in folder:
        output_dir=os.path.join(path, foldername)
        output_path=os.path.join(output_dir, setname)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with open(os.path.join(txtpath,setname+".txt"),"r") as f:
        
            while True:
                filelines = f.readline()
                if not filelines:
                    break
                    pass
                filename = filelines.strip('\n')
                if foldername=="json":
                    filename=filename.replace('.jpg', '_marked.json')
                filepath=os.path.join(dir, foldername,filename)
                if not os.path.isfile(filepath):
                    print ("%s not exist!"%filepath)
                else:
                    shutil.copy(filepath,output_path)
            print("%scompleted"%(foldername+setname))

