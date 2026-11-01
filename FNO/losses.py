from torch.nn import functional as F

def pde_res(u, Vp, Vs, density, dx=5.0, dt=0.001):
    _, C, T, W = u.shape
    nt = C // 2
    ux = u[:, :nt, :, :]
    uz = u[:, nt:, :, :]

    Vp_up      = F.interpolate(Vp,      size=(T, W), mode='bilinear', align_corners=False)
    Vs_up      = F.interpolate(Vs,      size=(T, W), mode='bilinear', align_corners=False)
    density_up = F.interpolate(density, size=(T, W), mode='bilinear', align_corners=False)

    ux_tt = (ux[:, :, 2:, :] - 2*ux[:, :, 1:-1, :] + ux[:, :, :-2, :]) / (dt**2)
    uz_tt = (uz[:, :, 2:, :] - 2*uz[:, :, 1:-1, :] + uz[:, :, :-2, :]) / (dt**2)

    ux_t = ux[:, :, 1:-1, :]
    uz_t = uz[:, :, 1:-1, :]

    def deriv_z(f):
        padded = F.pad(f, (1, 1, 0, 0), mode='constant', value=0)
        return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2 * dx)

    dux_dz = deriv_z(ux_t)
    duz_dz = deriv_z(uz_t)

    mu  = density_up * Vs_up ** 2
    lam = density_up * (Vp_up ** 2 - 2 * Vs_up ** 2)

    res_x = density_up[:, :, 1:-1, :] * ux_tt - (lam + 2*mu)[:, :, 1:-1, :] * deriv_z(dux_dz)[:, :, 1:-1, :]
    res_z = density_up[:, :, 1:-1, :] * uz_tt - (lam + 2*mu)[:, :, 1:-1, :] * deriv_z(duz_dz)[:, :, 1:-1, :]

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