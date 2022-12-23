# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
#-- for the server
import ray
from ray import serve

#-- for the client
import requests
import cv2
import matplotlib.pyplot as plt


# -

@serve.deployment(route_prefix="/image_rotate")
class ImageModel:
    def __init__(self):
        #do any initilization here, like loading a model!
        pass

    async def __call__(self, starlette_request):
        
        image_payload_bytes = await starlette_request.body()
        img=pickle.loads(image_payload_bytes)
        
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mean_val=img.mean()
        
        return pickle.dumps((img,mean_val))


# +
#ray.init()
# -

serve.start(address="auto", http_options={"host": "0.0.0.0"})
ImageModel.deploy()

imagedata = cv2.cvtColor(cv2.imread("09-322-02-1.bmp"),cv2.COLOR_BGR2RGB)
plt.imshow(imagedata)

resp=requests.post(url="http://127.0.0.1:8000/image_rotate",data=pickle.dumps(imagedata))

resp_output = pickle.loads(resp.content)
(returned_img,returned_mean_val)=resp_output

plt.imshow(returned_img)
print(returned_mean_val)


