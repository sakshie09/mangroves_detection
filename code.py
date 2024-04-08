import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import rasterio
import matplotlib.pyplot as plt

for i in range(4):
  with rasterio.open('/content/4april2020_2_nir.tiff') as src:
    img_NIR = src.read(1)
  with rasterio.open('/content/4april2020_2red.tiff') as src:
    img_RED = src.read(1)

ndvi = (img_NIR.astype(float) - img_RED.astype(float)) / (img_NIR.astype(float) + img_RED.astype(float))

# Finally we output these new pixel values to a new image file, making sure we mirror the GeoTIFF spatial metadata:
# Set spatial characteristics of the output object to mirror the input
kwargs = src.meta
kwargs.update(
    dtype=rasterio.float64,
    count = 1)

  # Create the file
with rasterio.open('ndvi.tif', 'w', **kwargs) as dst:
  dst.write_band(1, ndvi.astype(rasterio.float32))

norm = np.zeros((960, 1280))
final = cv2.normalize(ndvi,  norm, 0, 255, cv2.NORM_MINMAX)


cv2.imwrite("ndvi_cmap.png", final)
print(final)
plt.imshow(ndvi, cmap="Set1")
if i == 0:
  plt.colorbar()
  plt.savefig(f"{i}.png")

for i in range(4):
  with rasterio.open('/content/1jan2020_2_nir.tiff') as src:
    img_NIR = src.read(1)
  with rasterio.open('/content/1jan2020_2_red.tiff') as src:
    img_RED = src.read(1)

ndvi = (img_NIR.astype(float) - img_RED.astype(float)) / (img_NIR.astype(float) + img_RED.astype(float))

# Finally we output these new pixel values to a new image file, making sure we mirror the GeoTIFF spatial metadata:
# Set spatial characteristics of the output object to mirror the input
kwargs = src.meta
kwargs.update(
    dtype=rasterio.float64,
    count = 1)

plt.imshow(ndvi)

img = cv2.imread('/content/download.png')   # reads the image

grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2 reads iamge as bgr so we are converting it to rgb

grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)  #converting rgb image to hsv
lower_dark_pink = np.array([(159, 50, 70)])           #applying hsv range for pink as we have to extract the pink color out of the whole image
upper_dark_pink = np.array([180, 255, 255])

mask = cv2.inRange(grid_HSV, lower_dark_pink, upper_dark_pink)  # represents the elements of the two arrays representing the upper bounds and the lower bounds

res= cv2.bitwise_and(grid_RGB, grid_RGB,mask=mask)  #Computes the bit-wise AND of mask and rgb image element-wise
plt.imshow(res)

image = cv2.resize(res, (1200, 1200)) # resizing as per the resolution
plt.imshow(image)

img = cv2.imread('/content/May_HSV.png')
cv2_imshow(img)  #cv2 reads the image as gbr image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # converting bgr image to grayscale
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY) # ret is a boolean variable that returns true if the thresh is available
# The function cv.threshold is used to apply the thresholding. The first argument is the source image, which should be a grayscale image.
#The second argument is the threshold value which is used to classify the pixel values and
# third is maximum value which is assigned to pixel values exceeding the threshold.
area = cv2.countNonZero(thresh) # returns the number of nonzero pixels
print(area,'in pixels')
finall_area = area*8.784
print(finall_area,'in square meter')
