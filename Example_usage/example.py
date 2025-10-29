import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.fusion_with_modality import FusionStudent
from models.text_head_1 import TextClassifier
from models.distilhub import HubertSmall
#This loads the weights of each model that we used


text_model=TextClassifier(vocab_size=50260)
state_dict=torch.load(r"D:\modality_fusion\model_weights\text_model_weights.pt",map_location="cpu")
text_model.load_state_dict(state_dict,strict=False)


fusion = FusionStudent(text_ckpt_path=None, audio_ckpt_path=None)
fusion.text_model.student.resize_token_embeddings(50260)
fusion.load_state_dict(torch.load(r"D:\modality_fusion\model_weights\fusion_model_weights.pt"))

audio_model=HubertSmall()
state_dict=torch.load(r"D:\modality_fusion\model_weights\audio_model_weights.pt",map_location="cpu")
audio_model.load_state_dict(state_dict,strict=False)