import matplotlib.pyplot as plt
from skimage.restoration import inpaint

# Img = plt.imread('bird.png')
Img1 = (plt.imread('new2bird.png')*255).astype('uint8')
mask = Img1[:,:,0:3].sum(axis=-1)==255*3

# origin image
# plt.imshow(Img1[:,:,0:3])

# draw random orbit
# plt.imshow((Img1[:,:,0:3].sum(axis=-1)==255*3),cmap='gray')
Img_inpaint = inpaint.inpaint_biharmonic(Img1[:,:,0:3], mask, channel_axis=-1)
plt.imshow(Img_inpaint)
plt.show()