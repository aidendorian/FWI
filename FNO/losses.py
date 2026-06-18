from torch.nn import functional as F

def pde_res(u, Vp, Vs, density, dx=1., dt=1.):
    B, C, H, W = u.shape
    nt = C // 2
    ux = u[:, :nt, :, :]
    uz = u[:, nt:, :, :]

    Vp_up = F.interpolate(Vp, size=(H, W), mode='bilinear', align_corners=False)
    Vs_up = F.interpolate(Vs, size=(H, W), mode='bilinear', align_corners=False)
    density_up = F.interpolate(density, size=(H, W), mode='bilinear', align_corners=False)

    if Vp_up.dim() == 3:
        Vp_up = Vp_up.unsqueeze(1)
        Vs_up = Vs_up.unsqueeze(1)
        density_up = density_up.unsqueeze(1)

    ux_tt = (ux[:, 2:, :, :] - 2 * ux[:, 1:-1, :, :] + ux[:, :-2, :, :]) / (dt**2)
    uz_tt = (uz[:, 2:, :, :] - 2 * uz[:, 1:-1, :, :] + uz[:, :-2, :, :]) / (dt**2)

    ux_t = ux[:, 1:-1, :, :]
    uz_t = uz[:, 1:-1, :, :]

    def deriv_x(f):
        padded = F.pad(f, (0, 0, 1, 1), mode='constant', value=0)
        return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2 * dx)

    def deriv_z(f):
        padded = F.pad(f, (1, 1, 0, 0), mode='constant', value=0)
        return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2 * dx)

    dux_dx = deriv_x(ux_t)
    dux_dz = deriv_z(ux_t)
    duz_dx = deriv_x(uz_t)
    duz_dz = deriv_z(uz_t)

    mu = density_up * Vs_up ** 2
    lam = density_up * (Vp_up ** 2 - 2 * Vs_up ** 2)

    sigma_xx = (lam + 2 * mu) * dux_dx + lam * duz_dz
    sigma_zz = (lam + 2 * mu) * duz_dz + lam * dux_dx
    sigma_xz = mu * (dux_dz + duz_dx)

    div_sigma_x = deriv_x(sigma_xx) + deriv_z(sigma_xz)
    div_sigma_z = deriv_x(sigma_xz) + deriv_z(sigma_zz)

    crop = 2
    ux_tt_trim = ux_tt[:, :, crop:-crop, crop:-crop]
    uz_tt_trim = uz_tt[:, :, crop:-crop, crop:-crop]
    div_sigma_x_trim = div_sigma_x[:, :, crop:-crop, crop:-crop]
    div_sigma_z_trim = div_sigma_z[:, :, crop:-crop, crop:-crop]
    rho_trim = density_up[:, :, crop:-crop, crop:-crop]

    res_x = rho_trim * ux_tt_trim - div_sigma_x_trim
    res_z = rho_trim * uz_tt_trim - div_sigma_z_trim
    return (res_x ** 2 + res_z ** 2).mean()

def loss(pred_value,obs_value):
     mse_loss = F.mse_loss(pred_value,obs_value)
     return mse_loss

def FNO_loss(Vp,Vs,density,value_pred,value_obs,lambda_data=1.0,lambda_pde=0.1):

    L_data=loss(value_pred,value_obs)
    L_pde=pde_res(value_pred,Vp,Vs,density)
    weighted_data = L_data*lambda_data
    weighted_pde = L_pde*lambda_pde
    total_loss = weighted_data + weighted_pde
    return total_loss