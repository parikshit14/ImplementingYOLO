# ImplementingYOLO: Building a Custom Object Detector from Scratch

This repo let's you train a custom image detector using the state-of-the-art [YOLOv3](https://pjreddie.com/darknet/yolo/) computer vision algorithm.

###Basic Overview

To build and test your YOLO object detection algorithm follow the below steps:

 1. [Image Annotation](/1_Image_Annotation/)
	 - Install Microsoft's Visual Object Tagging Tool (VoTT)
	 - Annotate images
 2. [Training](/2_Training/)
 	- Download pre-trained weights
 	- Train your custom YOLO model on annotated images
 3. [Inference](/3_Inference/)
 	- Detect objects in new images and videos

## Repo structure

  + [`1_Image_Annotation`](/1_Image_Annotation/): Scripts and instructions on annotating images
  + [`2_Training`](/2_Training/): Scripts and instructions on training your YOLOv3 model
  + [`3_Inference`](/3_Inference/): Scripts and instructions on testing your trained YOLO model on new images and videos
  + [`Data`](/Data/): Input Data, Output Data, Model Weights and Results
  + [`Utils`](/Utils/): Utility scripts used by main scripts

## Model information
  + The model is trained on 960 images which had been annotated and augmented
  + The output classes containes **stone** , **paper** , **scissiors** .

## Getting Started

To run the detector function on the pretrained model:
 + Start by creating a virtual env if you have anaconda :
   ` conda create -n envname python==3.6.8 pip `
 + Setup a new virtual environment and install the depencencies from the requirements.txt files
   ` pip install -r requirements.txt `
 + Clone this repository
 + Run **detector.py** for using webcam and play the stone paper scissors game run this code :
   ` python 3_Inference/Detector.py ` considering that u are in the repo directory
 + Hope you enjoyed it, depends on the processor you are using the speed may varry
------------------------------------------------------
### Create your own Model

## Dataset:

To train the YOLO object detector on your own dataset, copy your training images to [`ImplementingYOLO/Data/Source_Images/Training_Images`](/Data/Source_Images/Training_Images/). By default, this directory is pre-populated few images and a folder. Feel free to delete all of them to make your project cleaner.
## Image Augmentation
To increase the number of images you can run the augmentor.py file which will increase the number of images by 8-9 times. this is an optional step just to increase the robustness and save your model from overfitting.
## Image Annotation
To make our detector learn, we first need to feed it some good training examples. We use Microsoft's Visual Object Tagging Tool (VoTT) to manually label images in our training folder [`ImplementingYOLO/Data/Source_Images/Training_Images`](/Data/Source_Images/Training_Images/). To achieve decent results annotate at least 100 images. For good results label at least 300 images and for great results label 1000+ images.
# Download VoTT
Head to VoTT [releases](https://github.com/Microsoft/VoTT/releases) and download and install the version for your operating system.

# Create a New Project
Create a **New Project** and call it `Annotations`. It is highly recommended to use `Annotations` as your project name. If you like to use a different name for your project, you will have to modify the command line arguments of subsequent scripts accordingly.

Under **Source Connection** choose **Add Connection** and put `Images` as **Display Name**. Under **Provider** choose **Local File System** and select [`ImplementingYOLO/Data/Source Images/Training_Images`](/Data/Source_Images/Training_Images) and then **Save Connection**. For **Target Connection** choose the same folder as for **Source Connection**. Hit **Save Project** to finish project creation.

![New Project](/1_Image_Annotation/Screen_Recordings/New_Project.gif)
# Export Settings
Navigate to **Export Settings** in the sidebar and then change the **Provider** to `Comma Separated Values (CSV)`, then hit **Save Export Settings**.

![New Project](/1_Image_Annotation/Screen_Recordings/Export_Settings.gif)
# Labeling
First create a new tag on the right and give it a relevant tag name. In our example, we choose `stone`, `paper`, `scissors`. Then draw bounding boxes around your objects. You can use the number key **no. key** to quickly assign the first tag to the current bounding box .
# Export Results
Once you have labeled enough images press **CRTL+E** to export the project. You should now see a folder called [`vott-csv-export`](/Data/Source_Images/Training_Images/vott-csv-export) in the [`Training_Images`](/Data/Source_Images/Training_Images) directory. Within that folder, you should see a `*.csv` file called [`Annotations-export.csv`](/Data/Source_Images/Training_Images/vott-csv-export/Annotations-export.csv) which contains file names and bounding box coordinates.
# Convert to YOLO Format
As a final step, convert the VoTT csv format to the YOLOv3 format. To do so, run the conversion script from within the [`ImplementingYOLO/1_Image_Annotation`](/1_Image_Annotation/) folder:
`python Convert_to_YOLO_format.py`
The script generates two output files:
**data_train.txt** in [`ImplementingYOLO/Data/Source_Images/Training_Images/vott-csv-export`](/Data/Source_Images/Training_Images/vott-csv-export)
**data_classes.txt** in [`ImplementingYOLO/Data/Model_Weights`](/Data/Model_Weights/)

## Training
Using the training images located in [`ImplementingYOLO/Data/Source_Images/Training_Images`](/Data/Source_Images/Training_Images) and the annotation file [`data_train.txt`](/Data/Source_Images/Training_Images/vott-csv-export) which we have created in the [previous step](/1_Image_Annotation/) we are now ready to train our YOLOv3 detector.

# Download and Convert Pre-Trained Weights
Before getting started download the pre-trained YOLOv3 weights and convert them to the keras format. To run both steps run the download and conversion script from within the [`ImplementingYOLO/2_Training`](/2_Training/) directory:
`python Download_and_Convert_YOLO_weights.py`
The weights are pre-trained on the [ImageNet 1000 dataset](http://image-net.org/challenges/LSVRC/2015/index) and thus work well for object detection tasks that are very similar to the types of images and objects in the ImageNet 1000 dataset.
# Train YOLOv3 Detector
To start the training, run the training script from within the [`ImplementingYOLO/2_Training`](/2_Training/) directory:
`python Train_YOLO.py `
Depending on your set-up, this process can take a few minutes to a few hours. The final weights are saved in [`ImplementingYOLO/Data/Model_weights`](/Data/Model_weights). It is highly recommended to use GPU for the training purpose it would be 30 times faster to train on it. Personally I used Google colab's Tesla k80 GPU for the training purpose. Time would be taken depending upon the size of the dataset, smaller the size lesser would be the time to train but low accuracy, robustness and vice versa.
**Attaching the colab link:-**[colab](https://colab.research.google.com/drive/1QCAFWiEcPyoNPUjK8N91kc0yA78_qco2?usp=sharing)
upload your customized dataset into the correct directory and few other files
train on this and after training download the **trained_weights_final.h5** and **trained_weights_stage_1.h5** and place them in the correct directory for the testing part.

## Testing
In this step, we test our detector on cat and dog images and videos located in [`ImplementingYOLO/Data/Source_Images/Test_Images`](/Data/Source_Images/Test_Images). If you like to test the detector on your own images or videos, place them in the [`Test_Images`](/Data/Source_Images/Test_Images) folder.

## Testing Your Detector
kindly rename detector(copy).py to detector.py for inference on your custom dataset
To detect objects run the detector script from within the [`ImplementingYOLO/3_Inference`](/3_Inference/) directory:.
`python Detector.py`
The outputs are saved to [`ImplementingYOLO/Data/Source_Images/Test_Image_Detection_Results`](/Data/Source_Images/Test_Image_Detection_Results). The outputs include the original images with bounding boxes and confidence scores as well as a file called [`Detection_Results.csv`](/Data/Source_Images/Test_Image_Detection_Results/Detection_Results.csv) containing the image file paths and the bounding box coordinates. For videos, the output files are videos with bounding boxes and confidence scores.

**__thats all, you have trained your own YOLO!!__**

- Please **star**  this repo to get notifications on future improvements and.
- Please **fork**  this repo if you like to use it as part of your own project.

__If facing any issues post it on the github i will try to resolve it as soon as possible.__






refered from->
```
joseph redmon and fadi ali paper on YOLO V3: [YOLOv3](https://pjreddie.com/darknet/yolo/)
```

```
@misc{TrainYourOwnYOLO,
  url={https://github.com/AntonMu/TrainYourOwnYOLO}
}
```
