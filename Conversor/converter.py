import torch
import torchvision.models as models
from torch.utils.mobile_optimizer import optimize_for_mobile

NUM_CLASSES = 11 

input_pkl_file = './ASDID/SqueezeNet-ASDID-91.70.pkl'
output_ptl_file = './modeloMobile/squeezenet_mobile.ptl'
model_architecture = models.squeezenet1_0(num_classes=NUM_CLASSES)

try:
    if torch.cuda.is_available():
        my_device = torch.device("cuda:0")
    else:
        my_device = torch.device("cpu")

    model = torch.load(input_pkl_file, map_location='cpu', weights_only=False) # carregar o modelo treinado
    
    model_architecture.eval() # definir o modelo para o modo de avaliação

    scripted_model = torch.jit.script(model_architecture) # converter o modelo para o formato TorchScript

    optimized_model = optimize_for_mobile(scripted_model) # otimizar o modelo para dispositivos móveis

    optimized_model._save_for_lite_interpreter(output_ptl_file) # salvar o modelo otimizado

except Exception as e:
    print(f"\nOcorreu um erro durante a conversão: {e}")