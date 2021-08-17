import numpy as np
import os

# 이것은 labeling 한 뒤, labeling이 안된 image를 삭제하는 코드입니다.
if __name__ == "__main__":
    location = './train_data/images/train'
    files = os.listdir(location)
    
    lists_img = []
    for name in files:
        sal = name.split('.')
        lists_img.append(sal[0])
    lists_img = set(lists_img)

    location = './train_data/labels/train'
    files = os.listdir(location)
    
    lists_label = []
    for name in files:
        sal = name.split('.')
        lists_label.append(sal[0])
    lists_label = set(lists_label)
    residual = lists_img.difference(lists_label)
    print("resudual: ", residual)

    for name in residual:
        text = name + '.jpg'
        os.remove('./train_data/images/train/' + text)
    print("finished!")
