# data_preprocessing
This repository contains the code for data preprocessing for the Capstone project.  
To run the code, please pull this repository and `traditional` repository to the same directory. The `traditional` repository can be found [here](https://github.com/MLforGPR/traditional).

## Overview of Steps
This section provides an overview of the steps required to generate the dataset. The detailed instructions are provided in the following sections.
1. Move the raw GPR data to the `unprocessed_images` folder in the root directory of the project.
2. Annotate the GPR data using the CVAT platform.
3. Export the annotations as a zip file.
4. Move the exported zip file to the `annotations` folder in the root directory of the project.
5. Run the `annotation_parser.ipynb` file to crop the images.
6. Run the `dataset_generator.ipynb` file to generate the dataset.

## Data
Create a folder named `unprocessed_images` in the root directory of the project, and the raw GPR data should be copied to this folder. The data should be saved in the following way:
```
root directory
|   annotation (annotation files)
│   unprocessed_images
│   │   WLT_350_210926__001 P_2111131.JPG
|   │   WLT_350_210926__003 P_2111131.JPG
|   │   ...
```

## Annotation
The annotation tool we used is [CVAT](https://www.cvat.ai/).
### CVAT Annotation Platform
CVAT is a powerful annotation platform that facilitates the labeling and review process. It provides a variety of tools to visualize and annotate data effectively. It also supports a wide range of annotation formats, including bounding boxes, polygons, polylines, points, and cuboids for 3D annotations. In this project, we use bounding boxes to annotate the data.

CVAT is open-source platform that can be deployed on-premise or in the cloud. For this project, we use the cloud version of CVAT. The cloud version of CVAT is free to use, making it a great choice for small projects.

CVAT also allows users to collaborate on projects and share annotated data. However, from our experience, using two different accounts to annotate the same data is not a good idea. The platform is not quite stable and it is easy to lose your work. Therefore, we recommend that you use one account to annotate the data, and share the login information with your teammates. Also, you should communicate with your teammates and ensure that no more than one user logs in to the platform at the same time.

### Annotation File Output
The output of the annotation process in CVAT is a zip file. This zip file contains all the annotated data, making it easy to manage and share your labeled datasets.

### Instructions
1. Accessing CVAT: Visit the CVAT website at https://cvat.org/ and create an account if you don't have one already.

2. Creating a Project: Once logged in, create a new project by clicking on the "New Project" button. Setup the project name and the name of every annotation.

3. Creating a Task and Uploading Data: Create a new Task and upload the GPR images you want to annotate. You can upload multiple images at once. Once the images are uploaded, you can start annotating them.

4. Annotating Objects: Use the selected annotation tool to annotate objects in the uploaded images.

5. Reviewing Annotations: After completing the annotation task, review the annotations to verify their correctness and consistency. Make any necessary adjustments or corrections as needed.

6. Saving Annotations: Save your annotations within the CVAT platform to preserve your work.

7. Exporting Annotations: After saving, you can export the annotations as a zip file. Click on the "Menu" button and select "Export job dataset" to export the annotations as a zip file. Choose the "CVAT for images 1.1" format and click on the "Export" button to export the annotations as a zip file. You can customize the name of the zip file. The zip file will be downloaded to your computer.

## Dataset Generation
The dataset generation contains two parts: cropping and dataset generation. The cropping process is implemented in the `annotation_parser.ipynb` file. The dataset generation process is implemented in the `dataset_generator.ipynb` and `dataset_generator_hyperbola.ipynb` file.

### Image Conversion
The `image_converter.ipynb` file takes the GPR image in PNG format as input and converts it to JPG format. The converted images are saved in the `unprocessed` folder in the root directory of the project.

### Image Cropping
The `annotation_parser.ipynb` file takes the zip file containing the annotations as input and generates the cropped image based on your setting. The cropped images are saved in the `unprocessed\cropped` folder in the root directory of the project.  
Move the exported zip file containing the annotations to the root directory of the project. 
The cropped images are saved in the following way:
```
root directory
│   annotation_parser.ipynb
|   train_1st2ndDataset.zip (exported zip file containing the annotations)
|   test_1st2ndDataset.zip
|   train_3rdDataset.zip
|   ...
│   unprocessed_images
|   |   cropped (cropped images)
|   |   │   │
|   |   │   └───200_40 (200x200 pixels and 40 pixel step)
|   |   │   |   |
|   |   │   |   └───train_1st2ndDataset (name of the CVAT annotations)
|   |   │   |   |   |
|   |   │   |   |   └───0 (image index of the dataset)
|   |   │   |   |   |   │   0_0.jpg (depth_index of the image)
|   |   │   |   |   |   │   0_1.jpg
|   |   │   |   |   |   │   ...
|   |   │   |   |   |   |   40_0.jpg
|   |   │   |   |   |   |   40_1.jpg
|   |   │   |   |   |   |   ...
|   |   |   |   |   |
|   |   │   |   |   └───1
|   |   │   |   |   |   │   0_0.jpg
|   |   │   |   |   |   |   ...
|   |   │   |   |   |
|   |   │   |   |   └───...
|   |   │   |   |
|   |   │   |   └───test_1st2ndDataset (name of the CVAT annotations)
|   |   │   |   |   │   ...
|   |   │   |   |
|   |   │   |   └───train_3rdDataset (name of the CVAT annotations)
```

### Dataset Generation
The `dataset_generator_hyperbola.ipynb` and `dataset_generator_layers.ipynb` file takes the cropped images as input and generates the dataset. The dataset is npz file containing the GPR data and the corresponding labels. The dataset is saved in the root directory of the project.  
The dataset generator used in final model is `dataset_generator_hyperbola.ipynb`.
```
root directory
|   dataset_generator_hyperbola.ipynb
|   dataset_generator_layers.ipynb
|   ...
```
