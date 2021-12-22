# coding: utf-8
import os
import random


def split(id_list,rate):
    random.shuffle(id_list)
    trainnum=int(rate[0]*n)
    testnum=int(rate[1]*n)
    valnum=int(rate[2]*n)
    train_id=id_list[:trainnum]
    test_id=id_list[trainnum:trainnum+testnum]
    val_id=id_list[trainnum+testnum:]
    return train_id,test_id,val_id


def find(id_list,classses):
    count=0
    i=0
    m=0
    for id in id_list:
        i+=1
        for filename in filelist:
            if id==num(filename):
                count+=1
                with open("/home/projectCodes/heart_point/data/"+classses+ '.txt', 'a+') as t:
                    t.writelines(filename+ '\n')
                t.close()
    print(i)
    print(count)


# def test(numslist):
#     with open('xxx.txt', "a+") as f:
#         for a in numslist:
#             f.writelines(a, '\n')
#     f.close()



def num(filename):
    file_str=list(filename) 
    for i,str in enumerate(file_str):
        if str.isalpha():
            break
    filenum=filename[:i]
    return  filenum


if __name__ == '__main__':
    data_dir="/home/projectCodes/dataprocess/images/"
    global filelist
    filelist=os.listdir(data_dir)
    id_list=[]
    for filename in filelist:
        filename=num(filename)
        id_list.append(filename)
    id_list=list(set(id_list))
    n=len(id_list)

    rate=[0.8,0.1,0.1]
    train_id,test_id,val_id=split(id_list,rate)

    find(train_id,"train")
    find(test_id,"test")
    find(val_id,"val")



        



