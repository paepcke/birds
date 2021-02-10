import random

# Functions bleew from https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
# Functions edited to support np arrays
def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.copy()
    num_mel_channels = cloned.shape[0]
    print(f"num_mel_channels is {num_mel_channels}")

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        print(f"Masked freq is [{f_zero} : {mask_end}]")
        if (replace_with_zero): cloned[f_zero:mask_end] = 0
        else: cloned[f_zero:mask_end] = cloned.mean()

    return cloned

def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.copy()
    len_spectro = cloned.shape[1]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        print(f"Masked time is [{t_zero} : {mask_end}]")
        if (replace_with_zero): cloned[:,t_zero:mask_end] = 0
        else: cloned[:,t_zero:mask_end] = cloned.mean()
        print(f"Mean inserted is {cloned.mean()}")
    return cloned
