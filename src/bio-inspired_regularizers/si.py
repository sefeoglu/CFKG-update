import torch

from transformers import Seq2SeqTrainer

class SISeq2SeqTrainer(Seq2SeqTrainer):
    
    def __init__(self, *args, si_lambda=1.0, epsilon=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_lambda = si_lambda
        self.epsilon = epsilon

        self.prev_params = {}
        self.omega = {}
        self.w = {}

        for n, p in self.model.named_parameters():
            self.prev_params[n] = p.clone().detach()
            self.omega[n] = torch.zeros_like(p)
            self.w[n] = torch.zeros_like(p)


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard loss
        outputs = model(**inputs)
        loss = outputs.loss

        # Add SI penalty
        si_penalty = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                delta = p - self.prev_params[n]
                si_penalty += (self.omega[n] * delta ** 2).sum()
        loss += self.si_lambda * si_penalty

        return (loss, outputs) if return_outputs else loss

    def training_step(self, *args, **kwargs):
        # After loss.backward(), accumulate importance
        loss = super().training_step(*args, **kwargs)

        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.w[n] -= p.grad.detach() * (p.detach() - self.prev_params[n])

        return loss

    def update_omega(self):
        """omega is used for surrogate loss"""
        for n, p in self.model.named_parameters():
            delta = p.detach() - self.prev_params[n]
            self.omega[n] += self.w[n] / (delta ** 2 + self.epsilon)
            self.prev_params[n] = p.clone().detach()
            self.w[n].zero_()