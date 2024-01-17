import cv2
import matplotlib.pyplot as plt

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

#The model for face detection
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

#Swapper
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

def faceswap(source, target):
    try: 
        source_image = cv2.imread(source)
        target_image = cv2.imread(target)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(source_image[:,:,::-1], aspect='equal')
        axs[0].axis('off')
        axs[0].set_title("source")
        axs[1].imshow(target_image[:,:,::-1], aspect='equal')
        axs[1].axis('off')
        axs[1].set_title("target")
        plt.show()

        
        face1 = app.get(source_image)[0]
        face2 = app.get(target_image)[0]
        swap = swapper.get(source_image, face1, face2, paste_back=True)

        plt.imshow(swap[:,:,::-1])
        plt.axis('off')
        plt.title("output")
        plt.show()
    except: 
        print("Error in processing images:". str(e))


print(" ")
print(" ")
print("faceswap starto")
img1 = input("input source image (body): ")
img2 = input("input target image (face): ")
faceswap(img1, img2)

