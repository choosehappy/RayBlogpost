import pickle
import cv2 
import ray
import requests
import torch
from ray import serve
import matplotlib.pyplot as plt

from torchvision.models import densenet201

@serve.deployment(
    route_prefix="/image_predict", ray_actor_options={"num_gpus": 1}
)
class ImageModel:
    def __init__(self):
        print(f"cuda:{ray.get_gpu_ids()[0]}")  
        self.device = torch.device(f"cuda:{ray.get_gpu_ids()[0]}")
        
        self.model = densenet201(pretrained=True) #this will downlaod the weights LOCALLY on each node for US
                                                  #if we were using our own model we would have to copy it
            
        self.model = self.model.to(self.device) #copy it to the GPU
        
        self.model.eval() #and put it in evaluation model


    async def __call__(self, starlette_request):

        image_payload_bytes = await starlette_request.body()
        img=pickle.loads(image_payload_bytes)
        
        
        img=img[None,:,:,:]
        gpu_array = torch.from_numpy(img.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(self.device)
        retval=self.model(gpu_array)
        retval = retval.detach().cpu().numpy()
        
        return pickle.dumps(retval)



ray.init()

serve.start(http_options={"host": "0.0.0.0"})

ImageModel.deploy()



# +
#---- Client Test
# -


imagedata = cv2.cvtColor(cv2.imread("09-322-02-1.bmp"),cv2.COLOR_BGR2RGB)
plt.imshow(imagedata)

# %%time
resp = requests.post("http://127.0.0.1:8000/image_predict", data=pickle.dumps(imagedata))

dl_output = pickle.loads(resp.content)

print(dl_output)

