import random
random.seed(0)
# Functions bleew from https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
# Functions edited to support np arrays
def freq_mask(spec, log, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.copy()
    num_mel_channels = cloned.shape[0]
    log.info(f"num_mel_channels is {num_mel_channels}")

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): continue

        mask_end = random.randrange(f_zero, f_zero + f)
        log.info(f"Masked freq is [{f_zero} : {mask_end}]")
        log.info(f"Mean inserted is {cloned.mean()}")
        if (replace_with_zero): cloned[f_zero:mask_end] = 0
        else: cloned[f_zero:mask_end] = cloned.mean()
    return cloned, f"fmask{int(f_zero)}_{int(mask_end)}"

def time_mask(spec, log, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.copy()
    len_spectro = cloned.shape[1]
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t):
            mask_end = 0
            continue

        mask_end = random.randrange(t_zero, t_zero + t)
        log.info(f"Masked time is [{t_zero} : {mask_end}]")
        log.info(f"Mean inserted is {cloned.mean()}")
        if (replace_with_zero): cloned[:,t_zero:mask_end] = 0
        else: cloned[:,t_zero:mask_end] = cloned.mean()
    return cloned, f"tmask{int(t_zero)}_{int(mask_end)}"
