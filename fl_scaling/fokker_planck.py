# FiPy model with our parameters
# Fokker-Planck Equation for the IOU process
# Our parameters from SW LIM ITER
#
# Roman Sokolovskii

from fipy import *
import numpy as np
import scipy.stats as stats
from tqdm import tnrange

#def sim_fokker_planck_iou_sc_ldpc(e, mean_ou, nu, theta, gstar, N, L, W, Iit, Iinit, Iinit_ones, scale, taustar_ppd_vn, vn_speed_pd, dt = 1, rho = 0.0, therange=None):
def sim_fokker_planck_iou_sc_ldpc(e, mean_ou, nu, theta, gstar, N, L, W, Iit, Iinit, scale, taustar_ppd_vn, vn_speed_pd, dt = 1, rho = 0.0, therange=None):
    wave_speed = vn_speed_pd
    init_period = taustar_ppd_vn
    # Iinit = Iinit - init_period

    # For starting from zero
    pure_ss_forming_time = init_period - 1 / mean_ou / wave_speed
    Iinit = Iinit - pure_ss_forming_time

    # # Initial distribution for IOU 
    # one_index = Iinit_ones
    # ones_mean = np.mean(one_index)
    # ones_variance = np.var(one_index)

    # ig_mu = ones_mean
    # ig_lambda = ig_mu ** 3 / ones_variance

    # bm_nu = 1 / ig_mu
    # bm_sigma = np.sqrt(1 / ig_lambda)
    # bm_t = taustar_ppd_vn
    # #bm_t = ig_mu
    # bm_loc = bm_nu * bm_t
    # bm_scale = bm_sigma * np.sqrt(bm_t)

    numPos = L
    num_iter = Iit * (numPos - 1) + Iinit

    sigma = np.sqrt(nu / N)
    tau = 1 / theta / (gstar - e)

    beta_ou = 1 / tau
    sigma_ou = sigma * np.sqrt(2. * beta_ou) * wave_speed
    mu_ou = mean_ou * wave_speed - 1 / Iit
    sigma_iou = scale

    # x - OU
    # y - IOU

    # OU parameters (specified above)
    # beta_ou = 2
    # sigma_ou = 1
    # mu_ou = 1   # Mean of the OU process
    D_ou = sigma_ou**2 / 2
    D_iou = sigma_iou**2 / 2

    sigma_ou_stationary = np.sqrt(sigma_ou**2 / 2 / beta_ou)
    Lx = 8 * sigma_ou_stationary # Length of domain x: +-4*sigma
    Ly = W     # Length of domain y
    Nx = 200   # Number of discretization points x
    Ny = Ly * 20   # Number of discretization points y
    dx = Lx / Nx
    dy = Ly / Ny

    cx = Lx / 2 - mu_ou  # center point for the grid
    #cy = 2
    #cy = 1 - Iinit / Iit
    cy = -(1 - Iinit / Iit)
    #print(cy, Iinit, pure_ss_forming_time)

    m = Grid2D(nx=Nx, ny=Ny, dx=dx, dy=dy) - ((cx,), (cy,))

    x = np.array(m.x[:Nx])
    y = np.array(m.y[::Nx])

    ## Subtracting dy/2 to have the value on the bottom grid
    ##p0 = ml.bivariate_normal(m.x, m.y - dy/2, sigma_ou_stationary, 1e-1, mu_ou, 1)
    #p0 = ml.bivariate_normal(m.x, m.y, sigma_ou_stationary, sigma_ou_stationary, mu_ou, 1)
    xy = np.dstack((m.x, m.y))

    # Initial condition
    #rv = stats.multivariate_normal(mean=[mu_ou, 1 + mu_ou + 1 / Iit], cov=[[sigma_ou_stationary**2, 0],[0, sigma_ou_stationary**2]])
    #rv = stats.multivariate_normal(mean=[mu_ou, bm_loc], cov=[[sigma_ou_stationary**2, 0],[0, bm_scale**2]])

    #iou_scale = bm_scale
    #rv = stats.multivariate_normal(mean=[mu_ou, bm_loc],\
    #                               cov=[[sigma_ou_stationary**2, rho * sigma_ou_stationary * iou_scale],\
    #                                    [rho * sigma_ou_stationary * iou_scale, iou_scale**2]]) 

    # Starting IOU from zero...
    iou_scale = 1e-1
    rv = stats.multivariate_normal(mean=[mu_ou, 0],\
                                   cov=[[sigma_ou_stationary**2, rho * sigma_ou_stationary * iou_scale],\
                                        [rho * sigma_ou_stationary * iou_scale, iou_scale**2]]) 
    p0 = rv.pdf(xy)
     
    p = CellVariable(mesh=m, value=p0)
    pval = np.reshape(p(m.cellCenters.globalValue), (Ny, Nx)).T
    ptotal_init = np.sum(pval) * dx * dy

    # Absorbing barrier at zero
    # (reflecting barrier at W is enforced by default)
    p.constrain(0, m.facesBottom)

    # Convection field for our IOU process.
    # Should be a vector (x, y) = (beta_ou*(x - mu_ou), -x)
    x_faceCoord = m.faceCenters[0]
    x_convection = beta_ou * (x_faceCoord - mu_ou)
    y_convection = -x_faceCoord #- dx/2  # to account for the differences in the grid...?
    xy_convection = [x_convection, y_convection]
    convection = FaceVariable(mesh=m, value=xy_convection)


    #D = [[[D_ou, 0], [0, 0]]]  # Diffusion only along X
    D = [[[D_ou, 0], [0, D_iou]]]  # Diffusion along both X and Y

    eq = TransientTerm() == ConvectionTerm(coeff=convection) + DiffusionTerm(D)

    #dt = 0.1 * 1./2 * dx
    #dt = 1 / 5
    #dt = 1

    num_time_points = int(num_iter / dt)
    results = []
    varmeans = []

    pval = np.reshape(p(m.cellCenters.globalValue), (Ny, Nx)).T
    px = np.sum(pval, axis=1) * dy
    py = np.sum(pval, axis=0) * dx
    results.append((pval, px, py))
    mx = np.sum(x * px) * dx
    my = np.sum(y * py) * dy
    varx = np.sum((x - mx)**2 * px) * dx
    vary = np.sum((y - my)**2 * py) * dy
    varmeans.append((mx, my, varx, vary))


    range_provided = False
    if therange is None:
        therange = range(num_time_points)
    else:
        range_provided = True
        therange = therange(num_time_points)

    for step in therange:
        # print('time step', step, 'out of', num_time_points)
        eq.solve(var=p, dt=dt)

        pval = np.reshape(p(m.cellCenters.globalValue), (Ny, Nx)).T
        px = np.sum(pval, axis=1) * dy
        py = np.sum(pval, axis=0) * dx
        ptotal = np.sum(pval) * dx * dy
        results.append((pval, px, py))

        # print("INTs: ", np.sum(px) * dx, np.sum(py) * dy)
        mx = np.sum(x * px) * dx
        my = np.sum(y * py) * dy
        # print("MEAN: ", mx, my)
        varx = np.sum((x - mx)**2 * px) * dx
        vary = np.sum((y - my)**2 * py) * dy
        # print("VARs: ", varx, vary)
        # print(beta_ou, sigma_ou, mu_ou)
        # print()
        varmeans.append((mx, my, varx, vary))
        if range_provided:
            therange.set_description("%.4e" % (ptotal_init - ptotal))

    fail_prob = ptotal_init - ptotal
    return fail_prob, results, varmeans
