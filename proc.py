import numpy

__ALL__ = [
    'create_target_model',
    'create_target_candidate',
    'calc_weight',
    'meanshift',
]

NUM_BIN = 8

_BAND = 255 /NUM_BIN

def to_b_num(img):
    """
    convert image to correpsonding bin index
    """
    assert isinstance(img, numpy.ndarray)
    b = (img //_BAND).astype(int)
    return numpy.minimum(b, 7)

def create_kernel(r, c):
    """
    create kernel with epancechnikov profile
    
    @param r, c: size
    """
    rr = numpy.arange(r) / (r-1) *2 -1
    cc = numpy.arange(c) / (c-1) *2 -1
    C, R = numpy.meshgrid(cc, rr)
    X2 = C**2 + R**2
    
    kernel = numpy.maximum(1-X2, 0)
    return kernel / numpy.sum(kernel)

def create_target_model(target_img):
    """
    create a target model pdf
    """
    B = to_b_num(target_img)
    
    kernel = create_kernel(*target_img.shape[:2])
    
    M = numpy.empty((NUM_BIN, 3))
    for b in range(NUM_BIN):
        for ch in range(3):
            M[b, ch] = numpy.sum(kernel[B[:,:,ch]==b])
            
    return M

def create_target_candidate(img, center, diameter):
    """
    create target candidate pdf
    
    @param img       the entire img
    @param center    center (x, y)
    @param diameter  mask diameter, aka normalized factor
    """
    R, C = img.shape[:2]
    
    c, r = center
    c = int(c)
    r = int(r)
    
    radius = int(min(diameter/2, R-r, C-c))
    
    target_img = img[r-radius:r+radius, c-radius:c+radius]
    return create_target_model(target_img)

def calc_weight(img, target_model, target_candidate, center, diameter):
    """
    calculate weight of each valid locaiton on target image
    based on Bhattacharyya Coefficient with target model
    """
    assert isinstance(center, numpy.ndarray)
    
    # pre-calculate weight coef
    C = numpy.zeros((NUM_BIN, 3))
    idx = target_candidate > 0
    C[idx] = numpy.sqrt(target_model[idx] / target_candidate[idx])
    
    # estimate range
    diameter = int(diameter)

    cnr_c, cnr_r = center.astype(int) - (diameter >> 1)
    target_img = to_b_num(img[cnr_r:cnr_r+diameter, cnr_c:cnr_c+diameter])

    #
    r, c, _ = target_img.shape
    weight = numpy.empty((r, c))
    for i in range(target_img.shape[0]):
        for j in range(target_img.shape[1]):
            b_val = target_img[i, j]
            
            c_val = 1.0
            for ch in range(3):
                c_val *= C[b_val[ch], ch]
            weight[i, j] = c_val
    
    # create G(X), derivation of kernel k(x)
    rr = numpy.arange(r) / (r-1) *2 -1
    cc = numpy.arange(c) / (c-1) *2 -1
    
    C, R = numpy.meshgrid(cc, rr)
    G = (C**2 + R**2) < 1
    
    # apply mask G(X)
    density = weight *G
    density /= numpy.sum(density)
    
    return density

def meanshift(density):
    """
    a round of meanshift in the algorithm
    
    @param  weight weight of the target candidates
    @return        translation, in (x, y)
    """
    # create G(X), derivation of kernel k(x)
    r, c = density.shape
    
    rr = numpy.arange(r) / (r-1) *2 -1
    cc = numpy.arange(c) / (c-1) *2 -1
    
    C, R = numpy.meshgrid(cc, rr)
    
    # measure shift
    C *= density
    R *= density
    
    sft_c = numpy.sum(C) *c
    sft_r = numpy.sum(R) *r
    
    return int(sft_c), int(sft_r)
