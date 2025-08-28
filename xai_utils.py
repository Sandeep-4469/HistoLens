import torch
import torch.nn.functional as F
import numpy as np

class BaseCAM:
    def __init__(self, model, target_layer):
        self.model, self.target_layer = model, target_layer; self.gradients, self.activations = None, None; self.hooks = []; self._register_hooks()
    def _register_hooks(self):
        self.hooks.append(self.target_layer.register_forward_hook(self._save_activations)); self.hooks.append(self.target_layer.register_full_backward_hook(self._save_gradients))
    def _save_activations(self, module, input, output): self.activations = output[0]
    def _save_gradients(self, module, grad_input, grad_output): self.gradients = grad_output[0]
    def remove_hooks(self): [h.remove() for h in self.hooks]
    def calculate_cam(self): raise NotImplementedError

class GradCAM(BaseCAM):
    def calculate_cam(self):
        if self.gradients is None or self.activations is None: raise RuntimeError("Gradients/Activations not found for GradCAM.")
        image_grads = self.gradients[:, 1:, :]; image_acts = self.activations[:, 1:, :]
        weights = torch.mean(image_grads, dim=1); cam = torch.einsum('bpd,bd->bp', image_acts, weights)
        cam = cam.squeeze().cpu().to(torch.float32).detach().numpy(); cam = np.maximum(cam, 0)
        return cam / (cam.max() + 1e-8)

class GradCAMpp(BaseCAM):
    def calculate_cam(self):
        if self.gradients is None or self.activations is None: raise RuntimeError("Gradients/Activations not found for GradCAM++.")
        image_grads = self.gradients[:, 1:, :]; image_acts = self.activations[:, 1:, :]
        grads_pos = F.relu(image_grads); alpha_num = grads_pos.pow(2)
        alpha_denom = 2 * grads_pos.pow(2) + torch.sum(image_acts * grads_pos.pow(3), dim=-1, keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-8); weights = torch.sum(alpha * grads_pos, dim=-1)
        cam = torch.einsum('bp,bpd->bd', weights, image_acts).sum(dim=-1)
        cam = cam.squeeze().cpu().to(torch.float32).detach().numpy(); cam = np.maximum(cam, 0)
        return cam / (cam.max() + 1e-8)
        
class HiResCAM(BaseCAM):
    def calculate_cam(self):
        if self.gradients is None or self.activations is None: raise RuntimeError("Gradients/Activations not found for HiResCAM.")
        image_grads = self.gradients[:, 1:, :]; image_acts = self.activations[:, 1:, :]
        cam = (image_acts * image_grads).sum(dim=-1)
        cam = cam.squeeze().cpu().to(torch.float32).detach().numpy(); cam = np.maximum(cam, 0)
        return cam / (cam.max() + 1e-8)

class GuidedBackprop:
    def __init__(self, model):
        self.model, self.hooks = model, []; self._override_relu()
    def _override_relu(self):
        def relu_backward_hook(module, grad_in, grad_out): return (F.relu(grad_in[0]),)
        for module in self.model.vision_tower.modules():
            if isinstance(module, torch.nn.ReLU): self.hooks.append(module.register_full_backward_hook(relu_backward_hook))
    def remove_hooks(self): [h.remove() for h in self.hooks]
    def calculate_gradients(self, pixel_values):
        if pixel_values.grad is None: raise RuntimeError("Guided Backprop failed: pixel_values.grad is None.")
        grad = pixel_values.grad.cpu().numpy()[0]
        return np.transpose(grad, (1, 2, 0))