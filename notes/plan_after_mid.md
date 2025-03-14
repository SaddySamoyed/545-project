后续计划: 

1. automate the procedure for (relatively) large dataset (e.g. 3 million images)

   this needs: the balance between efficiency and quality of inpainting. So we may need to tweak the tech stack and architecture slightly after practice

2. further optimize the structure: if a class of images are already well-represented, then we do not need to improve the accuracy of shotting concerning images.

   for example: if there are already lots of very similar images as this image in the dataset, then we need not decompose it.

   We need to reselect a subset of all images in a dataset, only to do our decomposition on these images. This is for efficiency.