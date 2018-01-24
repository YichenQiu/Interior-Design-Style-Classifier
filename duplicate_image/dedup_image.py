from PIL import Image
import imagehash
import os
from collections import defaultdict



def dedup_image(directories):
    for label in directories[:1]:
        d=defaultdict(list)
        for image in os.listdir('Interior-Design-Style-Classifier/{}'.format(label)):
            im=Image.open('Interior-Design-Style-Classifier/{}/{}'.format(label,image))
            h=str(imagehash.dhash(im))
            d[h]+=[image]
        lst=[]
        for k,v in d.items():
            if len(v)>1:
                lst.append(list(v))
        for item in lst:
            for image in item [1:]:
                os.unlink("Interior-Design-Style-Classifier/{}/{}".format(label,image))

if __name__==__main__:
    directories=['Bohemian','Coastal','Industrial','Scandinavian']
    dedup_image(directories)
