import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from utils.fusion import SDFM

class DiTWithDINOInjection(nn.Module):
    def __init__(self, device, dino_path, weight_dtype=torch.bfloat16, dit_text_dim=3584):
        super().__init__()
        
        self.device = device
        self.weight_dtype = weight_dtype
        self.dino_path = dino_path
        print(f"Loading DINOv3 from: {self.dino_path}")
        
        self.img_processor = AutoImageProcessor.from_pretrained(self.dino_path)
        self.img_encoder_model = AutoModel.from_pretrained(self.dino_path)
        
        def img_encoder(pixel_values):
            outputs = self.img_encoder_model(pixel_values=pixel_values)
            return outputs.last_hidden_state
        
        self.img_encoder = img_encoder
        
        self.fusion = SDFM(
            token_dim=1024,
            hidden_dim=512,
            use_layernorm=True
        )
        
        print(f"DiT text dimension: {dit_text_dim}, DINO dimension: 1024")
        if dit_text_dim != 1024:
            print(f"Adding projection layer: 1024 -> {dit_text_dim}")
            self.dino_projection = nn.Linear(1024, dit_text_dim)
        else:
            print("No projection needed, dimensions match!")
            self.dino_projection = None
        
        self.img_encoder_model.to(self.device, dtype=self.weight_dtype)
        self.fusion.to(self.device, dtype=self.weight_dtype)
        if self.dino_projection is not None:
            self.dino_projection.to(self.device, dtype=self.weight_dtype)
        
        self.fusion.train()
        if self.dino_projection is not None:
            self.dino_projection.train()
        
        self.img_encoder_model.eval()
        self.img_encoder_model.requires_grad_(False)
    
    @staticmethod
    def _get_dit_text_dim(generator):
        try:
            dit = generator.pipe.dit
            if hasattr(dit, 'txt_norm'):
                if hasattr(dit.txt_norm, 'normalized_shape'):
                    dim = dit.txt_norm.normalized_shape[0]
                    print(f"Detected from txt_norm.normalized_shape: {dim}")
                    return dim
                elif hasattr(dit.txt_norm, 'weight'):
                    dim = dit.txt_norm.weight.shape[0]
                    print(f"Detected from txt_norm.weight: {dim}")
                    return dim
            
            if hasattr(dit, 'txt_in'):
                if hasattr(dit.txt_in, 'in_features'):
                    dim = dit.txt_in.in_features
                    print(f"Detected from txt_in.in_features: {dim}")
                    return dim
                elif hasattr(dit.txt_in, 'linear'):
                    if hasattr(dit.txt_in.linear, 'in_features'):
                        dim = dit.txt_in.linear.in_features
                        print(f"Detected from txt_in.linear.in_features: {dim}")
                        return dim
                    elif hasattr(dit.txt_in.linear, 'weight'):
                        dim = dit.txt_in.linear.weight.shape[1]
                        print(f"Detected from txt_in.linear.weight: {dim}")
                        return dim
            
            print("Using Qwen standard text dimension: 3584")
            return 3584
            
        except Exception as e:
            print(f"Error detecting text dimension: {e}")
            print("Falling back to Qwen standard: 3584")
            return 3584
    
    @torch.no_grad()
    def extract_dino_features(self, images):
        if not isinstance(images, list):
            images = [images]
        
        pixel_values = self.img_processor(
            images=images, 
            return_tensors="pt"
        )["pixel_values"].to(device=self.device, dtype=self.weight_dtype)
        
        features = self.img_encoder(pixel_values)
        return features
    
    def forward(self, lq_image, sr_image):
        with torch.no_grad():
            lr_tokens = self.extract_dino_features(lq_image)
            sr_tokens = self.extract_dino_features(sr_image)
        
        prompt_embeds, alpha = self.fusion(sr_tokens, lr_tokens)
        
        if self.dino_projection is not None:
            prompt_embeds = self.dino_projection(prompt_embeds)
        
        B, seq_len, _ = prompt_embeds.shape
        prompt_emb_mask = torch.ones(B, seq_len, dtype=torch.bool, device=self.device)
        
        return prompt_embeds, prompt_emb_mask, alpha
    
    def save_checkpoint(self, save_dir, iteration):
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        fusion_path = os.path.join(save_dir, f"fusion-{iteration}.pth")
        torch.save(self.fusion.state_dict(), fusion_path)
        print(f"Saved fusion weights to: {fusion_path}")
        
        if self.dino_projection is not None:
            proj_path = os.path.join(save_dir, f"projection-{iteration}.pth")
            torch.save(self.dino_projection.state_dict(), proj_path)
            print(f"Saved projection weights to: {proj_path}")
    
    def load_checkpoint(self, load_dir, iteration=None):
        import os
        import glob
        
        if iteration is not None:
            fusion_path = os.path.join(load_dir, f"fusion-{iteration}.pth")
        else:
            fusion_files = glob.glob(os.path.join(load_dir, "fusion-*.pth"))
            if not fusion_files:
                raise FileNotFoundError(f"No fusion checkpoint found in {load_dir}")
            fusion_path = sorted(fusion_files)[-1] 
        
        if not os.path.exists(fusion_path):
            raise FileNotFoundError(f"Fusion checkpoint not found: {fusion_path}")
        
        print(f"Loading fusion weights from: {fusion_path}")
        state = torch.load(fusion_path, map_location="cpu")
        
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}
        
        self.fusion.load_state_dict(state, strict=True)
        print("✓ Fusion weights loaded successfully!")
        
        if self.dino_projection is not None:
            if iteration is not None:
                proj_path = os.path.join(load_dir, f"projection-{iteration}.pth")
            else:
                proj_files = glob.glob(os.path.join(load_dir, "projection-*.pth"))
                if proj_files:
                    proj_path = sorted(proj_files)[-1]
                else:
                    proj_path = None
            
            if proj_path and os.path.exists(proj_path):
                print(f"Loading projection weights from: {proj_path}")
                proj_state = torch.load(proj_path, map_location="cpu")
                if isinstance(proj_state, dict) and "state_dict" in proj_state:
                    proj_state = proj_state["state_dict"]
                self.dino_projection.load_state_dict(proj_state, strict=True)
                print("✓ Projection weights loaded successfully!")
