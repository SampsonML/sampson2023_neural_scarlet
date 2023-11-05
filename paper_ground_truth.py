import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import random, jit
from numpyro.distributions import constraints
import scarlet2
from scarlet2 import *
import matplotlib.pyplot as plt
import pandas as pd
from scarlet2 import nn
from scarlet2 import relative_step
from functools import partial
import btk
import scarlet 
import cmasher as cmr
from matplotlib.colors import LogNorm
from galaxygrad import HSC_ScoreNet64, ZTF_ScoreNet64, HSC_ScoreNet32
import cmasher as cmr

plt.rcParams["xtick.top"] = True 
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = 'in' 
plt.rcParams["ytick.direction"] = 'in' 
plt.rcParams["xtick.minor.visible"] = True 
plt.rcParams["ytick.minor.visible"] = True 
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["xtick.minor.size"] = 4.5
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["ytick.minor.size"] = 4.5
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["xtick.minor.width"] = 1.5
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["ytick.minor.width"] = 1.5
plt.rcParams['axes.linewidth'] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({"text.usetex": True})
#plt.style.use('dark_background')

def flux_comparison(seed, max_n_sources, stamp_size=12, max_shift=2, spec_weight=1, band=2):

    catalog = btk.catalog.CatsimCatalog.from_file('input_catalog.fits')
    sampling_function = btk.sampling_functions.DefaultSampling(
        max_number=max_n_sources, min_number=max_n_sources, # always 3 sources in every blend.
        stamp_size=stamp_size, 
        max_shift=max_shift, # shift is only 2 arcsecs = 10 pixels, which means blends are likely.
        min_mag = -np.inf, max_mag = 26.5, # magnitude range of the galaxies
        seed = seed)
    LSST = btk.survey.get_surveys('LSST')

    batch_size = 1

    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog,
        sampling_function,
        LSST,
        batch_size=batch_size,
        stamp_size=stamp_size,
        njobs=1,
        add_noise="all",
        seed=seed, # use same seed here
    )

    # run the scarlet deblending
    blend_batch = next(draw_generator)
    deblender = btk.deblend.Scarlet(max_n_sources,max_components=1) 
    deblend_ex = deblender.deblend(0, blend_batch, reference_catalogs=blend_batch.catalog_list)
    deblend_batch = deblender(blend_batch, njobs=1, reference_catalogs=blend_batch.catalog_list)

    # grab images
    x_centers = blend_batch.catalog_list[0]["x_peak"]
    y_centers = blend_batch.catalog_list[0]["y_peak"]
    centers = np.stack([x_centers, y_centers], axis=1)

    # get the defaul scarlet source initialisation
    # initialize scarlet
    image = blend_batch.blend_images[0]
    n_bands = image.shape[0]
    psf = blend_batch.get_numpy_psf()
    wcs = blend_batch.wcs
    survey = blend_batch.survey
    bands = survey.available_filters
    cat = blend_batch.catalog_list[0]
    img_size = blend_batch.image_size
    # get background
    #filters = [survey.get_filter(band) for band in bands]
    #bkg = np.array([mean_sky_level(survey, f).to_value("electron") for f in filters])

    model_psf = scarlet.GaussianPSF(sigma=(0.7,) * n_bands)
    model_frame = scarlet.Frame(image.shape, psf=model_psf, channels=bands, wcs=wcs)
    model_frame_scar = scarlet.Frame(image.shape, psf=model_psf, channels=bands, wcs=wcs)
    scarlet_psf = scarlet.ImagePSF(psf)
    weights = np.ones(image.shape) 
    obs = scarlet.Observation(image, psf=scarlet_psf, weights=weights, channels=bands, wcs=wcs)
    observations = obs.match(model_frame)
    ra_dec = np.array([cat["ra"] / 3600, cat["dec"] / 3600]).T

    thresh = 1.0
    e_rel  = 1e-5
    max_iter = 200
    min_snr = 50

    sources, _ = scarlet.initialization.init_all_sources(
                    model_frame,
                    ra_dec,
                    observations,
                    max_components=1,
                    min_snr=min_snr,
                    thresh=thresh,
                    fallback=True,
                    silent=True,
                    set_spectra=True,
                )

    # alter spectrum values
    for i in range(len(sources[0].parameters[0])):
        sources[0].parameters[0][i] = sources[0].parameters[0][i] * spec_weight

    morph_init = [None]*len(sources)
    for k, src in enumerate(sources):
        morph_init[k] = np.array(src.parameters[1])
        
    spec_init = [None]*len(sources)
    for k, src in enumerate(sources):
        spec_init[k] = np.array(src.parameters[0])

    blend = scarlet.Blend(sources, observations)
    it, logL = blend.fit(200, e_rel=1e-4)

    individual_sources = []
    for component in sources:
        y, x = component.center
        model = component.get_model(frame=model_frame)
        model_ = observations.render(model)
        individual_sources.append(model_)
    deblended_images = np.zeros((max_n_sources, n_bands, img_size, img_size))
    deblended_images[: len(individual_sources)] = individual_sources
    deblended_scarlet1 = deblended_images

    # set up scarlet2 deblender
    images = blend_batch.blend_images[0]
    weights = jnp.ones(image.shape) 
    #weights = jnp.zeros(image.shape)
    frame_psf = GaussianPSF(0.7)
    psf = blend_batch.get_numpy_psf()

    # get centers
    x_centers = blend_batch.catalog_list[0]["x_peak"]
    y_centers = blend_batch.catalog_list[0]["y_peak"]
    centers = np.stack([y_centers, x_centers], axis=1)
    center2 = [src.center for src in sources]

    # box and center params
    model_frame = Frame(
                    Box(images.shape), 
                    psf=frame_psf)
    obs = Observation(images, weights, psf=ArrayPSF(jnp.asarray(psf)))
    obs.match(model_frame);

    # load the model you wish to use
    prior_model = HSC_ScoreNet64
    prior_model = ZTF_ScoreNet64

    def transform(x):
        sigma_y = 0.10
        return jnp.log(x + 1) / sigma_y

    keys = random.split(random.PRNGKey(0), 2)
    spec_step = partial(relative_step, factor=1e-2)
    morph_step = partial(relative_step, factor=1e-1)
    with Scene(model_frame) as scene:
        for i in range(len( centers )):
            # define new prior here for each new model
            prior = nn.ScorePrior(model=prior_model, transform=transform, model_size=64)
            Source(
                centers[i],
                ArraySpectrum(Parameter(spec_init[i], 
                                        constraint=constraints.positive, 
                                        stepsize=spec_step)),
                ArrayMorphology(Parameter(morph_init[i],
                                        prior=prior, 
                                        constraint=constraints.positive,
                                        stepsize=1e-2))
            )
            
    # now fit the model
    scene_fitted = scene.fit(obs, max_iter=200, e_rel=2e-4);
    renders = obs.render(scene_fitted())

    def get_extent(bbox):
        return [bbox.start[-1], bbox.stop[-1], bbox.start[-2], bbox.stop[-2]]

    def normalize(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    model_scarlet2 = scene_fitted.sources[0]()
    renders_scarlet2 = obs.render(model_scarlet2)

    band_idx = band # user selected
    truths = blend_batch.isolated_images[:, :, band_idx][0]
    scarlets = deblended_scarlet1[:,band_idx,:,:]

    # initialise residuals
    resid_1 = np.zeros(truths.shape[0])
    resid_2 = np.zeros(truths.shape[0])
    true_flux = np.zeros(truths.shape[0])
    corr_1 = np.zeros(truths.shape[0])
    corr_2 = np.zeros(truths.shape[0])

    for i in range(truths.shape[0]):
        resid_1_tmp    = 0
        resid_2_tmp    = 0
        true_flux_tmp  = 0
        
        corr_1_tmp    = 0
        corr_2_tmp    = 0
        
        for j in range(band):
            band_idx = 1
            truths = blend_batch.isolated_images[:, :, band_idx][0]
            scarlets = deblended_scarlet1[:,band_idx,:,:]
            scarlet1_spec = np.array(sources[i].parameters[0])
            
            srcs = scene_fitted.sources
            scarlet2_spec = np.array(srcs[i].spectrum())
            center = np.array(srcs[i].morphology.bbox.center)[::-1]
            renders = obs.render(srcs[i]())[band_idx]
            extent = get_extent(obs.frame.bbox)
            # no frame2box routine in scarlet2 so lts do this explicitly here
            scene_frame = np.zeros(obs.frame.bbox.shape)[0]
            #scene_frame = scarlet2.plot.img_to_rgb(scene_frame, channel_map=None)
            small_image = renders  #scarlet2.plot.img_to_rgb(renders, channel_map=None)

            # Calculate the extent for the rendered model 
            extent_render = [center[0] - renders.shape[0] // 2,
                    center[0] + renders.shape[0] // 2,
                    center[1] - renders.shape[1] // 2,
                    center[1] + renders.shape[1] // 2]
            pad_x_low = np.abs(np.min([0, extent[0] - extent_render[0]])) #-1 
            pad_x_high = np.abs(np.min([0, extent_render[1] - extent[1]])) -1
            pad_y_low = np.abs(np.min([0, extent[2] - extent_render[2]])) #-1
            pad_y_high = np.abs(np.min([0, extent_render[3] - extent[3]])) -1
            test = np.pad(small_image, [(pad_y_low, pad_y_high), (pad_x_low, pad_x_high)], mode='constant', constant_values=0)

            resid_1_tmp    = np.sum(scarlets[i] - truths[i])
            resid_2_tmp    = np.sum(test - truths[i])
            true_flux_tmp  = np.sum((truths[i]))
            
            corr_1_tmp    = np.dot(truths[i].flatten(), scarlets[i].flatten()) / np.dot(np.sqrt(np.dot(truths[i].flatten(), truths[i].flatten())) , np.sqrt(np.dot(scarlets[i].flatten(), scarlets[i].flatten())))
            corr_2_tmp    = np.dot(truths[i].flatten(), test.flatten()) / np.dot(np.sqrt(np.dot(truths[i].flatten(), truths[i].flatten())) , np.sqrt(np.dot(test.flatten(), test.flatten())))

        
            
        resid_1[i]   = resid_1_tmp
        resid_2[i]   = resid_2_tmp
        true_flux[i] = true_flux_tmp
        
        corr_1[i]   = corr_1_tmp
        corr_2[i]   = corr_2_tmp
        
        #spec1 = np.dot(truths[i].flatten(), scarlets[i].flatten()) / np.dot(np.sqrt(np.dot(truths[i].flatten(), truths[i].flatten())) , np.sqrt(np.dot(scarlets[i].flatten(), scarlets[i].flatten())))
        #spec2 = 
        
        
        
    return resid_1, resid_2, true_flux, corr_1, corr_2 


max_n_sources = [6, 7, 8]
stamp_size    = 24
num_trials    = 10
seed          = np.linspace(1,num_trials,num_trials) 
max_shift     = 2.4
spec_weight   = 1

resid_scarlet_1 = []
resid_scarlet_2 = []
true_flux_all   = []

corr_scarlet_1 = []
corr_scarlet_2 = []

import os
#os.system('clear')

for seed_n in seed:
    n_sources = 6
    try:
        resid_1, resid_2, true_flux, corr1, corr2 =  flux_comparison(int(seed_n), 
                                                       n_sources, 
                                                       stamp_size, 
                                                       max_shift,
                                                       spec_weight,
                                                       band=1)
        
        resid_scarlet_1 = np.append(resid_scarlet_1, resid_1)
        resid_scarlet_2 = np.append(resid_scarlet_2, resid_2)
        true_flux_all = np.append(true_flux_all, true_flux)
        corr_scarlet_1 = np.append(corr_scarlet_1, corr1)
        corr_scarlet_2 = np.append(corr_scarlet_2, corr2)
    except:
        print(f'failed at seed {seed_n}')
    #clear_output(wait=True)
    os.system('clear')
    print(f'finished trial: {int(seed_n)} / {int(seed[-1])}')


from scipy.stats import gaussian_kde
# Calculate the point density
x = np.log10(true_flux_all)
y = resid_scarlet_1 / true_flux_all
mask = ~np.isnan(y)
x = x[mask]
y = y[mask]
xy = np.vstack([x,y])
z1 = gaussian_kde(xy)(xy)

x2 = np.log10(true_flux_all)
y2 = resid_scarlet_2 / true_flux_all
mask = ~np.isnan(y2)
x2 = x2[mask]
y2 = y2[mask]
xy = np.vstack([x2,y2])
z2 = gaussian_kde(xy)(xy)

x3 = np.log10(true_flux_all)
y3 = corr_scarlet_1
mask = ~np.isnan(y2)
x3 = x3[mask]
y3 = y3[mask]
xy = np.vstack([x3,y3])
z3 = gaussian_kde(xy)(xy)

x4 = np.log10(true_flux_all)
y4 = corr_scarlet_2
mask = ~np.isnan(y4)
x4 = x4[mask]
y4 = y4[mask]
xy = np.vstack([x4,y4])
z4 = gaussian_kde(xy)(xy)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 13),dpi=100)
cMAP = 'cmr.iceburn'
m_size = 20

plt.subplot(2,2,1)
plt.scatter(x, y, 
            c = z1, 
            cmap=cMAP,
            s=m_size,
            alpha=0.95,
            label = r'$\textsc{Scarlet}2$')
#plt.hist2d(x, y, bins=200, cmap='cmr.iceburn', density=False)
plt.hlines(0, -100, 100, color='r', linestyle='--',linewidth=2.5)
plt.xlim(0.95 * np.min(np.log10(true_flux_all)), 1.05 * np.max(np.log10(true_flux_all)))
plt.ylim(-.75, .75)
plt.xticks(fontsize=0)
plt.yticks(fontsize=14)
plt.text(5.5, 0.6, r'$\textsc{Scarlet}1$', fontsize=21, color='k')
#plt.xlabel(r'$\log_{10}$(true flux)',fontsize=25)
plt.ylabel('(model - true) / true',fontsize=25)

# now scarlet 2
plt.subplot(2,2,2)
im1 = plt.scatter(x2, y2, 
            c = z2, 
            cmap=cMAP,
            s=m_size,
            alpha=0.95,
            label = r'$\textsc{Scarlet}2$')
plt.hlines(0, -100, 100, color='r', linestyle='--',linewidth=2.5)
plt.ylim(-.75, .75)
plt.text(5.5, 0.6, r'$\textsc{Scarlet}2$', fontsize=21, color='k')
plt.xlim(0.95 * np.min(np.log10(true_flux_all)), 1.05 * np.max(np.log10(true_flux_all)))
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
#plt.xlabel(r'$\log_{10}$(true flux)',fontsize=25)

plt.subplot(2,2,3)
plt.hlines(1, -100, 100, color='r', linestyle='--',linewidth=2.5)
plt.scatter(x3, y3, 
            c = z3, 
            cmap=cMAP,
            s=m_size,
            alpha=0.95,
            label = r'$\textsc{Scarlet}2$')
plt.xlim(0.95 * np.min(np.log10(true_flux_all)), 1.05 * np.max(np.log10(true_flux_all)))
plt.ylim(.5, 1.1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\log_{10}$(true flux)',fontsize=24)
plt.ylabel('morphology correlation',fontsize=24)

# now scarlet 2
plt.subplot(2,2,4)
plt.hlines(1, -100, 100, color='r', linestyle='--',linewidth=2.5)
plt.yticks(fontsize=0)
plt.xticks(fontsize=14)
im = plt.scatter(x4, y4, 
            c = z4, 
            cmap=cMAP,
            s=m_size,
            alpha=0.95,
            label = r'$\textsc{Scarlet}2$')
plt.ylim(.5, 1.1)
plt.xlim(0.95 * np.min(np.log10(true_flux_all)), 1.05 * np.max(np.log10(true_flux_all)))
plt.xlabel(r'$\log_{10}$(true flux)',fontsize=24)

plt.subplots_adjust(wspace=.0, hspace=.05)
#plt.colorbar(im, orientation="vertical",fraction=0.07,anchor=(5.0,0.0))
cbar_ax = fig.add_axes([0.9, 0.11, 0.02, 0.3755])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label(r'number of sources',fontsize=24)

cbar_ax2 = fig.add_axes([0.9, 0.504, 0.02, 0.376])
cbar2 = plt.colorbar(im1, cax=cbar_ax2)
cbar2.set_label(r'number of sources',fontsize=24)

plt.savefig('flux_comparison_1000_alt.pdf', bbox_inches='tight',dpi=200)
plt.savefig('flux_comparison_1000_alt.png', bbox_inches='tight',dpi=200)

name = 'flux_comparison_plot' + str(num_trials) + '.pdf'
name2 = 'flux_comparison_plot' + str(num_trials) + '.png'
plt.savefig(name, bbox_inches='tight',dpi=200)
plt.savefig(name2, bbox_inches='tight',dpi=200)

PREFIX = '/Users/mattsampson/Research/Melchior/scarlet_development/'
NAME   = 'flux_test_' + str(num_trials) + '_'
SUFFIX =  str(num_trials) + '.npy'
np.save(PREFIX + NAME + 'x' + SUFFIX, x, allow_pickle=True)
np.save(PREFIX + NAME + 'y' + SUFFIX, y , allow_pickle=True)
np.save(PREFIX + NAME + 'x2' + SUFFIX, x2, allow_pickle=True)
np.save(PREFIX + NAME + 'y2' + SUFFIX, y2 , allow_pickle=True)
np.save(PREFIX + NAME + 'x3' + SUFFIX, x3, allow_pickle=True)
np.save(PREFIX + NAME + 'y3' + SUFFIX, y3 , allow_pickle=True)
np.save(PREFIX + NAME + 'x4' + SUFFIX, x4, allow_pickle=True)
np.save(PREFIX + NAME + 'y4' + SUFFIX, y4 , allow_pickle=True)

loaded = True
if loaded:
    x = np.load(PREFIX + NAME + 'x' + SUFFIX)
    y = np.load(PREFIX + NAME + 'y' + SUFFIX)
    x2 = np.load(PREFIX + NAME + 'x2' + SUFFIX)
    y2 = np.load(PREFIX + NAME + 'y2' + SUFFIX)
    x3 = np.load(PREFIX + NAME + 'x3' + SUFFIX)
    y3 = np.load(PREFIX + NAME + 'y3' + SUFFIX)
    x4 = np.load(PREFIX + NAME + 'x4' + SUFFIX)
    y4 = np.load(PREFIX + NAME + 'y4' + SUFFIX)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 13),dpi=100)
cMAP = 'cmr.iceburn'
m_size = 20

plt.subplot(2,2,1)
plt.scatter(x, y, 
            c = z1, 
            cmap=cMAP,
            s=m_size,
            alpha=0.95,
            label = r'$\textsc{Scarlet}2$')
#plt.hist2d(x, y, bins=200, cmap='cmr.iceburn', density=False)
plt.hlines(0, -100, 100, color='r', linestyle='--',linewidth=2.5)
plt.xlim(0.95 * np.min(np.log10(true_flux_all)), 1.05 * np.max(np.log10(true_flux_all)))
plt.ylim(-.75, .75)
plt.xticks(fontsize=0)
plt.yticks(fontsize=14)
plt.text(5.1, 0.6, r'$\textsc{Scarlet}1$', fontsize=21, color='k')
#plt.xlabel(r'$\log_{10}$(true flux)',fontsize=25)
plt.ylabel('(model - true) / true',fontsize=25)

# now scarlet 2
plt.subplot(2,2,2)
im1 = plt.scatter(x2, y2, 
            c = z2, 
            cmap=cMAP,
            s=m_size,
            alpha=0.95,
            label = r'$\textsc{Scarlet}2$')
plt.hlines(0, -100, 100, color='r', linestyle='--',linewidth=2.5)
plt.ylim(-.75, .75)
plt.text(5.1, 0.6, r'$\textsc{Scarlet}2$', fontsize=21, color='k')
plt.xlim(0.95 * np.min(np.log10(true_flux_all)), 1.05 * np.max(np.log10(true_flux_all)))
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
#plt.xlabel(r'$\log_{10}$(true flux)',fontsize=25)

plt.subplot(2,2,3)
plt.hlines(1, -100, 100, color='r', linestyle='--',linewidth=2.5)
plt.scatter(x3, y3, 
            c = z3, 
            cmap=cMAP,
            s=m_size,
            alpha=0.95,
            label = r'$\textsc{Scarlet}2$')
plt.xlim(0.95 * np.min(np.log10(true_flux_all)), 1.05 * np.max(np.log10(true_flux_all)))
plt.ylim(.5, 1.1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\log_{10}$(true flux)',fontsize=24)
plt.ylabel('morphology correlation',fontsize=24)

# now scarlet 2
plt.subplot(2,2,4)
plt.hlines(1, -100, 100, color='r', linestyle='--',linewidth=2.5)
plt.yticks(fontsize=0)
plt.xticks(fontsize=14)
im = plt.scatter(x4, y4, 
            c = z4, 
            cmap=cMAP,
            s=m_size,
            alpha=0.95,
            label = r'$\textsc{Scarlet}2$')
plt.ylim(.5, 1.1)
plt.xlim(0.95 * np.min(np.log10(true_flux_all)), 1.05 * np.max(np.log10(true_flux_all)))
plt.xlabel(r'$\log_{10}$(true flux)',fontsize=24)

plt.subplots_adjust(wspace=.0, hspace=.05)
#plt.colorbar(im, orientation="vertical",fraction=0.07,anchor=(5.0,0.0))
cbar_ax = fig.add_axes([0.9, 0.11, 0.02, 0.3755])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label(r'number of sources',fontsize=24)

cbar_ax2 = fig.add_axes([0.9, 0.504, 0.02, 0.376])
cbar2 = plt.colorbar(im1, cax=cbar_ax2)
cbar2.set_label(r'number of sources',fontsize=24)

name = 'flux_comparison_plot_loaded' + str(num_trials) + '.pdf'
name2 = 'flux_comparison_plot_loaded' + str(num_trials) + '.png'
plt.savefig(name, bbox_inches='tight',dpi=200)
plt.savefig(name2, bbox_inches='tight',dpi=200)
