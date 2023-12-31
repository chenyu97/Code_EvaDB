import torch
from torch import nn
import torch.nn.functional as F
from .backbones.CLIP import clip 
from torchvision.models import resnet50 
from .backbones.senet import se_resnext50_32x4d
from .backbones.efficientnet import EfficientNet
from .backbones import open_clip

supported_img_encoders = ["open_clip", "CLIP","resnet50","se_resnext50_32x4d","efficientnet-b0","efficientnet-b2","efficientnet-b3"]
supported_lang_encoders = ["open_clip", "CLIP","BERT"]

'''
For Text-Image Retrieval with Image and Language Encoder are CLIP

'''
class CLIP_recognition(nn.Module):
    def __init__(self, model_cfg):
        super(CLIP_recognition, self).__init__()

        # visual feature extractor
        self.model_cfg = model_cfg
        self.embed_dim  = self.model_cfg.EMBED_DIM
        if self.model_cfg.IMG_ENCODER in  supported_img_encoders:
            #print(f"====> Using visual backbone: {self.model_cfg.IMG_ENCODER}")
            if self.model_cfg.IMG_ENCODER == "CLIP":
                #print(f"====> Using CLIP type: {self.model_cfg.CLIP_TYPE}")
                if self.model_cfg.CLIP_TYPE == "open_clip":
                    self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
                else:
                    self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda", jit=False)
                self.clip_model = self.clip_model.float()
                self.clip_dim = 512
                self.crop_model = self.clip_model.encode_image
                self.bg_model = self.crop_model	
                self.projection_bg = nn.Linear(self.clip_dim , self.embed_dim)
                self.projection_crop = nn.Linear(self.clip_dim , self.embed_dim)

            elif self.model_cfg.IMG_ENCODER == "resnet50":
                self.crop_model = resnet50(pretrained=False,
                                        num_classes=model_cfg.OUTPUT_SIZE)
                state_dict = torch.load(self.model_cfg.RESNET_CHECKPOINT,
                                        map_location=lambda storage, loc: storage.cpu())
                del state_dict["fc.weight"]
                del state_dict["fc.bias"]
                self.crop_model.load_state_dict(state_dict, strict=False)	
                self.bg_model = self.crop_model
                self.img_in_dim = 1024
                self.projection_bg = nn.Sequential(nn.ReLU(),nn.Linear(self.img_in_dim, self.embed_dim))
                self.projection_crop = nn.Sequential(nn.ReLU(),nn.Linear(self.img_in_dim, self.embed_dim))

            elif self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.crop_model = se_resnext50_32x4d()
                self.bg_model = self.crop_model	
                self.img_in_dim = 2048
                self.projection_bg = nn.Conv2d(self.img_in_dim, self.embed_dim,kernel_size=1)
                self.projection_crop = nn.Conv2d(self.img_in_dim, self.embed_dim,kernel_size=1)
            
            else:
                self.crop_model = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.bg_model = self.crop_model	
                self.img_in_dim = self.crop_model.out_channels
                self.projection_bg = nn.Linear(self.img_in_dim, self.embed_dim)
                self.projection_crop = nn.Linear(self.img_in_dim, self.embed_dim)
        
        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"

        #projection head 
        self.projection_head_lang = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.projection_head_bg = nn.Sequential(nn.BatchNorm1d(self.embed_dim),nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim//2))
        self.projection_head_crop = nn.Sequential(nn.BatchNorm1d(self.embed_dim),nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim//2))
        self.projection_head_visual = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.BatchNorm1d(self.embed_dim),nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.id_cls = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.model_cfg.NUM_CLASSES))
        self.id_cls_crop = nn.Sequential(nn.Linear(self.embed_dim//2, self.embed_dim),nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.model_cfg.NUM_CLASSES))
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)

    def visual_forward(self, crop, bg):
        crop_embeds = self.projection_crop(self.crop_model(crop).float())
        crop_embeds = crop_embeds.view(crop_embeds.size(0), -1) 
        crop_head = self.projection_head_crop(crop_embeds)

        bg_embeds = self.projection_bg(self.bg_model(bg).float())
        bg_embeds = bg_embeds.view(bg_embeds.size(0), -1)   
        bg_head = self.projection_head_bg(bg_embeds)

        merge_head = self.projection_head_visual(torch.cat([bg_head,crop_head],dim=-1))
        merge_head = F.normalize(merge_head, p = 2, dim = -1)
        return merge_head  

    def crop_visual_forward(self, crop):
        crop_embeds = self.projection_crop(self.crop_model(crop).float())
        crop_embeds = crop_embeds.view(crop_embeds.size(0), -1) 
        crop_head = self.projection_head_crop(crop_embeds)
        crop_head = F.normalize(crop_head, p = 2, dim = -1)
        return crop_head  

    def forward(self, crop, bg):
        
        if self.model_cfg.ONLY_CROP:
            visual_head = self.crop_visual_forward(crop)
            cls_logits_visual = self.id_cls_crop(visual_head)
        else:
            visual_head = self.visual_forward(crop,bg)
            cls_logits_visual = self.id_cls(visual_head)

        return cls_logits_visual
