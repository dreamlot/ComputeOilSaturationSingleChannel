# ComputeOilSaturationSingleChannel
Compute the bright-phase saturation in segmented images of a single channel of two phases. The image dataset is published on Digital Rocks Portal https://www.digitalrocksportal.org/projects/367.

cut.py cuts images in a directory to the wanted size.
imshow.py implements cv2.imshow().
split_images.py splits the images along horizontal direction.
compute_So_whole computes the oil (bright phase) saturation in the unsplitted image. It detects the upper and lower boundary of the flow channel and counts the number of bright pixels in between.
compute_So_whole computes the oil (bright phase) saturation in the splitted image. It detects the upper and lower boundary of the flow channel and counts the number of bright pixels in between.




