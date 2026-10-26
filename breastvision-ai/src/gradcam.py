from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from .engine import MEAN, STD

class GradCAM:
    """Small Grad-CAM implementation targeting the final EfficientNet feature block."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]
    def _save_activation(self, module, inputs, output):
        self.activations = output
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    def __call__(self, input_tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[:, class_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam, logits.detach()
    def close(self):
        for handle in self.handles:
            handle.remove()

def preprocess_pil(image: Image.Image, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

def overlay_gradcam(image: Image.Image, cam: np.ndarray, alpha=0.38):
    import matplotlib.cm as cm
    base = image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    base_np = np.asarray(base).astype(np.float32) / 255.0
    heat = cm.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip((1 - alpha) * base_np + alpha * heat, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))
