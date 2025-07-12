import torch

def gradient_penalty(Discriminator, real_samples, fake_samples):
    """Compute Gradient Penalty loss

    Args:
        Discriminator : Discriminator model
        real_samples : Real earth data from the dataset
        fake_samples : Generated data from the Generator
    """
    batch_size = real_samples.shape[0]
    alpha = torch.randn(batch_size, 1, 1, 1, device=real_samples.device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    d_interpolated = Discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    
def discrminator_loss(Discriminator, real_samples, fake_samples, lambda_gp=10.0):
    d_real = Discriminator(real_samples).mean()
    d_fake = Discriminator(fake_samples).mean()
    gp = gradient_penalty(Discriminator, real_samples, fake_samples)
    loss = d_fake - d_real + lambda_gp*gp
    return loss