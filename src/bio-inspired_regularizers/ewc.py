import torch
from transformers import Seq2SeqTrainer
def freeze_old_params(model):
    return {name: param.detach().clone() for name, param in model.named_parameters()}

def compute_fisher(model, dataloader):
    """
    Compute the Fisher Information matrix for a given model and dataloader.
    Assumes model outputs a dict with a 'loss' key.
    """

    # Step 1: Initialize Fisher information dictionary
    fisher = {
        name: torch.zeros_like(param, device=param.device)
        for name, param in model.named_parameters() if param.requires_grad
    }

    # Step 2: Set model to evaluation mode
    model.eval()

    # Step 3: Accumulate squared gradients
    with torch.no_grad(): # Keep no_grad here as gradients are computed within the loop
        total_samples = 0
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
                loss = outputs.loss
                loss.backward()

                # Accumulate squared gradients, checking if name exists in fisher
                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad and name in fisher:
                        fisher[name] += param.grad.detach() ** 2

            total_samples += 1

    # Step 4: Average Fisher values
    for name in fisher:
        fisher[name] /= total_samples

    return fisher
class EWCTrainer(Seq2SeqTrainer):
    def __init__(self, *args, fisher=None, old_params=None, ewc_lambda=0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.fisher = fisher
        self.old_params = old_params
        self.ewc_lambda = ewc_lambda

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute standard supervised loss
        outputs = model(**inputs)
        loss = outputs.loss

        # Add EWC loss if fisher and old_params are provided
        if self.old_params is not None and self.fisher is not None:
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in self.fisher:
                    fisher_val = self.fisher[name]
                    old_param = self.old_params[name]
                    ewc_loss += (fisher_val * (param - old_param).pow(2)).sum()
            loss = loss + self.ewc_lambda * ewc_loss

        return (loss, outputs) if return_outputs else loss
