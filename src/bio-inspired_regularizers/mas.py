
from transformers import Seq2SeqTrainer
import torch

class MASTrainer(Seq2SeqTrainer):
    def __init__(self, *args, old_params=None, importance=None, mas_lambda=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_params = old_params
        self.importance = importance
        self.mas_lambda = mas_lambda

    # Add num_items_in_batch to the method signature
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Standard loss
        outputs = model(**inputs)
        loss = outputs.loss

        # MAS regularization
        if self.old_params and self.importance:
            mas_reg = 0.0
            print("MAS")
            for name, param in model.named_parameters():
                if name in self.old_params:
                    mas_reg += (self.importance[name] * (param - self.old_params[name]).pow(2)).sum()
            loss += self.mas_lambda * mas_reg

        return (loss, outputs) if return_outputs else loss
    
def freeze_old_params(model):
    return {name: param.detach().clone() for name, param in model.named_parameters()}

def compute_mas_importance(model, dataloader, device):
    importance = {}
    model.eval()

    # Initialize importance dict
    for name, param in model.named_parameters():
        importance[name] = torch.zeros_like(param)
    with torch.no_grad():

      for batch in dataloader:
          model.zero_grad()

            # Move batch to the same device as model
          inputs = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

            # Re-enable gradient computation for Fisher estimation
          for param in model.parameters():
              param.requires_grad = True

            # Forward pass with grad
          with torch.enable_grad():
              outputs = model(**inputs)
              loss = outputs.loss.norm(2)
              loss.backward()

                # Accumulate squared gradients, checking if name exists in fisher
              for name, param in model.named_parameters():
                if param.grad is not None:
                    importance[name] += param.grad.abs().detach()

    # Normalize
    for name in importance:
        importance[name] /= len(dataloader)

    return importance
