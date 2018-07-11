
## Table of contents

- [Motivation](#metrics-for-object-detection)
- [Different competitions, different metrics](#different-competitions-different-metrics)
- [Important definitions](#important-definitions)
- [Metrics](#metrics)
  - [Precision x Recall curve](#precision-x-recall-curve)
  - [Average Precision](#average-precision)
- [**How to use this project**](#how-to-use-this-project)
- [References](#references)

<a name="different-competitions-different-metrics"></a> 
## Different competitions, different metrics 

* **[PASCAL VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)** : The official documentation explaining their criteria for object detection metrics can be accessed [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000). The current metrics used by the current PASCAL VOC object detection challenge are the **Precision x Recall curve** and **Average Precision**.  

* **[COCO Detection Challenge](https://competitions.codalab.org/competitions/5181)** uses different metrics to evaluate the accuracy of object detection of different algorithms. [Here](http://cocodataset.org/#detection-eval) you can find a documentation explaining the 12 metrics used for characterizing the performance of an object detector on COCO. This competition offers Python and Matlab codes so users can verify their scores before submitting the results. 
## Put the image here//

* **[Google Open Images Dataset V4 Competition](https://storage.googleapis.com/openimages/web/challenge.html)** also uses mean Average Precision (mAP) over the 500 classes to evaluate the object detection task. 

* **[ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge)** defines an error for each image considering the class and the overlapping region between ground truth and detected boxes. The total error is computed as the average of all min errors among all test dataset images. [Here](https://www.kaggle.com/c/imagenet-object-localization-challenge#evaluation) are more details about their evaluation method.  

## Important definitions  

### Intersection Over Union (IOU)

Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between two bounding boxes. It requires a ground truth bounding box ![](http://latex.codecogs.com/gif.latex?B_%7Bgt%7D) and a predicted bounding box ![](http://latex.codecogs.com/gif.latex?B_p). By applying the IOU we can tell if a detection is valid (True Positive) or not (False Positive).  
IOU is given by the overlapping area between the predicted bounding box and the ground truth bounding box divided by the area of union between them:  

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?%5Ctext%7BIOU%7D%20%3D%20%5Cfrac%7B%5Ctext%7Barea%7D%20%5Cleft%28B_p%20%5Ccap%20B_%7Bgt%7D%5Cright%29%7D%7B%5Ctext%7Barea%7D%20%5Cleft%28B_p%20%5Ccup%20B_%7Bgt%7D%5Cright%29%7D">
</p>

The image below illustrates the IOU between a ground truth bounding box (in green) and a detected bounding box (in red).

<!--- IOU --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/iou.png" align="center"/></p>

### True Positive, False Positive, False Negative and True Negative  

Some basic concepts used by the metrics:  

* **True Positive (TP)**: A correct detection. Detection with IOU ≥ _threshold_  
* **False Positive (FP)**: A wrong detection. Detection with IOU < _threshold_  
* **False Negative (FN)**: A ground truth not detected  
* **True Negative (TN)**: Does not apply. It would represent a corrected misdetection. In the object detection task there are many possible bounding boxes that should not be detected within an image. Thus, TN would be all possible bounding boxes that were corrrectly not detected (_so many possible boxes within an image_). That's why it is not used by the metrics.

_threshold_: depending on the metric, it is usually set to 50%, 75% or 95%.

### Precision

Precision is the ability of a model to identify **only** the relevant objects. It is the percentage of correct positive predictions and is given by:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?Precision%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FP%7D%3D%5Cfrac%7BTP%7D%7B%5Ctext%7Ball%20detections%7D%7D">
</p>

### Recall 

Recall is the ability of a model to find all the relevant cases **(all ground truth bounding boxes)**. It is the percentage of true positive detected among all relevant ground truths and is given by:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?Recall%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FN%7D%3D%5Cfrac%7BTP%7D%7B%5Ctext%7Ball%20ground%20truths%7D%7D">
</p>

## Metrics

In the topics below there are some comments on the most popular metrics used for object detection.

### Precision x Recall curve

The Precision x Recall curve is a good way to evaluate the performance of an object detector as the confidence is changed. There is a curve for _each object class_. An object detector of a particular class is considered good if its **prediction stays high as recall increases**, which means that if you vary the confidence threshold, the precision and recall will still be high. This statement can be more intuitively understood by looking at the above eqautions of P and R and keeping in mind that **TP+FN = all ground truth = constant**, so Recall increases, means TP increased, hence FN will decrease. As TP has increased, only if FP decreases, will the Precision remain high i.e. the _model is doing less mistakes_ and hence is good.  Another way to identify a good object detector is to look for a detector that can identify only relevant objects (0 False Positives = high precision), finding all ground truth objects (0 False Negatives = high recall).  

A poor object detector **needs to increase the number of detected objects** (increasing False Positives = lower precision) in order to retrieve all ground truth objects (high recall). That's why the Precision x Recall curve usually starts with high precision values, _decreasing_ as recall increases. You can see an example of the Prevision x Recall curve in the next topic (Average Precision).  
This kind of curve is used by the PASCAL VOC 2012 challenge.  

### Average Precision

Another way to compare the performance of object detectors is to calculate the area under the curve (AUC) of the Precision x Recall curve. As AP curves are often zigzag curves frequently going up and down, comparing different curves (different detectors) in the same plot usually is not an easy task - because the curves tend to cross each other much frequently. That's why Average Precision (AP), a numerical metric, can also help us compare different detectors. In practice AP is the precision averaged across all recall values between 0 and 1.  

PASCAL VOC 2012 challenge uses the **interpolated average precision**. It tries to summarize the shape of the Precision x Recall curve by averaging the precision at a set of eleven equally spaced recall levels [0, 0.1, 0.2, ... , 1]:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cfrac%7B1%7D%7B11%7D%5Csum_%7Br%5Cin%5C%7B0%2C0.1%2C...%2C1%5C%7D%7D%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28r%5Cright%20%29%7D">
</p>

with

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28r%5Cright%20%29%7D%3D%5Cmax_%7B%5Cwidetilde%7Br%7D%3A%5Cwidetilde%7Br%7D%5Cgeqslant%7Br%7D%7D%20%5Crho%20%5Cleft%20%28%5Cwidetilde%7Br%7D%20%5Cright%29">
</p>

where ![](http://latex.codecogs.com/gif.latex?%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29) is the measured precision at recall ![](http://latex.codecogs.com/gif.latex?%5Ctilde%7Br%7D).

Instead of using the precision observed at each point, the AP is obtained by interpolating the precision at each level ![](http://latex.codecogs.com/gif.latex?r) taking the **maximum precision whose recall value is greater than ![](http://latex.codecogs.com/gif.latex?r)**.

Note that only the predictions(the red boxes below) are marked as TP or FP.     
#### An ilustrated example 

An example helps us understand better the concept of the interpolated average precision. Consider the detections below:
  
<!--- Image samples 1 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/samples_1_v2.png" align="center"/></p>
  
There are 7 images with 15 ground truth objects representented by the green bounding boxes and 24 detected objects represented by the red bounding boxes. Each detected object has a confidence level and is identified by a letter (A,B,...,Y).  

The following table shows the bounding boxes with their corresponding confidences. The last column identifies the detections as TP or FP. In this example a **TP is considered** if IOU ![](http://latex.codecogs.com/gif.latex?%5Cgeq) 30%, **otherwise it is a FP**. By looking at the images above we can roughly tell if the detections are TP or FP.

<!--- Table 1 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/table_1_v2.png" align="center"/></p>

<!---
| Images | Detections | Confidences | TP or FP |
|:------:|:----------:|:-----------:|:--------:|
| Image 1 | A | 88% | FP |
| Image 1 | B | 70% | TP |
| Image 1 |	C	| 80% | FP |
| Image 2 |	D	| 71% | FP |
| Image 2 |	E	| 54% | TP |
| Image 2 |	F	| 74% | FP |
| Image 3 |	G	| 18% | TP |
| Image 3 |	H	| 67% | FP |
| Image 3 |	I	| 38% | FP |
| Image 3 |	J	| 91% | TP |
| Image 3 |	K	| 44% | FP |
| Image 4 |	L	| 35% | FP |
| Image 4 |	M	| 78% | FP |
| Image 4 |	N	| 45% | FP |
| Image 4 |	O	| 14% | FP |
| Image 5 |	P	| 62% | TP |
| Image 5 |	Q	| 44% | FP |
| Image 5 |	R	| 95% | TP |
| Image 5 |	S	| 23% | FP |
| Image 6 |	T	| 45% | FP |
| Image 6 |	U	| 84% | FP |
| Image 6 |	V	| 43% | FP |
| Image 7 |	X	| 48% | TP |
| Image 7 |	Y	| 95% | FP |
--->

Note that, in some images there are **more than one detection overlapping a ground truth that are TP** (Images 2, 3, 4, 5, 6 and 7). For those cases the detection with the _highest IOU_ is taken, discarding the other detections. This rule is applied by the PASCAL VOC 2012 metric: "**e.g:** 5 detections (TP) of a single object is counted as 1 correct detection and 4 false detections”.

The Precision x Recall curve is plotted by calculating the precision and recall values of the accumulated TP or FP detections. For this, first we need to order the detections by their confidences, then we calculate the precision and recall for each accumulated detection as shown in the table below:       
**Note**: Total gt boxes = 15. So, recall will always be calculated as **(Acc TP)/15** in this case.

<!--- Table 2 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/table_2_v2.png" align="center"/></p>

<!---
| Images | Detections | Confidences |  TP | FP | Acc TP | Acc FP | Precision | Recall |
|:------:|:----------:|:-----------:|:---:|:--:|:------:|:------:|:---------:|:------:|
| Image 5 |	R	| 95% | 1 | 0 | 1 | 0 | 1       | 0.0666 |
| Image 7 |	Y	| 95% | 0 | 1 | 1 | 1 | 0.5     | 0.6666 |
| Image 3 |	J	| 91% | 1 | 0 | 2 | 1 | 0.6666  | 0.1333 |
| Image 1 | A | 88% | 0 | 1 | 2 | 2 | 0.5     | 0.1333 |
| Image 6 |	U	| 84% | 0 | 1 | 2 | 3 | 0.4     | 0.1333 |
| Image 1 |	C	| 80% | 0 | 1 | 2 | 4 | 0.3333  | 0.1333 |
| Image 4 |	M	| 78% | 0 | 1 | 2 | 5 | 0.2857  | 0.1333 |
| Image 2 |	F	| 74% | 0 | 1 | 2 | 6 | 0.25    | 0.1333 |
| Image 2 |	D	| 71% | 0 | 1 | 2 | 7 | 0.2222  | 0.1333 |
| Image 1 | B | 70% | 1 | 0 | 3 | 7 | 0.3     | 0.2    |
| Image 3 |	H	| 67% | 0 | 1 | 3 | 8 | 0.2727  | 0.2    |
| Image 5 |	P	| 62% | 1 | 0 | 4 | 8 | 0.3333  | 0.2666 |
| Image 2 |	E	| 54% | 1 | 0 | 5 | 8 | 0.3846  | 0.3333 |
| Image 7 |	X	| 48% | 1 | 0 | 6 | 8 | 0.4285  | 0.4    |
| Image 4 |	N	| 45% | 0 | 1 | 6 | 9 | 0.7     | 0.4    |
| Image 6 |	T	| 45% | 0 | 1 | 6 | 10 | 0.375  | 0.4    |
| Image 3 |	K	| 44% | 0 | 1 | 6 | 11 | 0.3529 | 0.4    |
| Image 5 |	Q	| 44% | 0 | 1 | 6 | 12 | 0.3333 | 0.4    |
| Image 6 |	V	| 43% | 0 | 1 | 6 | 13 | 0.3157 | 0.4    |
| Image 3 |	I	| 38% | 0 | 1 | 6 | 14 | 0.3    | 0.4    |
| Image 4 |	L	| 35% | 0 | 1 | 6 | 15 | 0.2857 | 0.4    |
| Image 5 |	S	| 23% | 0 | 1 | 6 | 16 | 0.2727 | 0.4    |
| Image 3 |	G	| 18% | 1 | 0 | 7 | 16 | 0.3043 | 0.4666 |
| Image 4 |	O	| 14% | 0 | 1 | 7 | 17 | 0.2916 | 0.4666 |
--->
 
 Plotting the precision and recall values we have the following *Precision x Recall curve*:
 
 <!--- Precision x Recall graph --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/precision_recall_example_1_v2.png" align="center"/>
</p>
 
As seen before, the idea of the **interpolated average precision** is to average the precisions at a set of 11 recall levels (0,0.1,...,1). The interpolated precision values are obtained by taking the maximum precision whose recall value is greater than its current recall value. We can visually obtain those values by looking at the recalls starting from the highest (0.4666) to 0 (looking at the plot from right to left) and, as we decrease the recall, we annotate the precision values that are the highest as shown in the image below:

<!--- interpolated precision curve --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/interpolated_precision_v2.png" align="center"/>
</p>

The Average Precision (AP) is the AUC obtained by the interpolated precision. The intention is to reduce the impact of the wiggles in the Precision x Recall curve. We divide the AUC into 4 areas (A1, A2, A3 and A4) as shown below:
  
<!--- interpolated precision AUC --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/interpolated_precision-AUC_v2.png" align="center"/>
</p>

Calculating the total area, we have the AP:  

![](http://latex.codecogs.com/gif.latex?AP%20%3D%20A1%20&plus;%20A2%20&plus;%20A3%20&plus;%20A4)  
  
![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bwith%3A%7D)  
![](http://latex.codecogs.com/gif.latex?A1%20%3D%20%280.0666-0%29%5Ctimes1%20%3D%5Cmathbf%7B0.0666%7D)  
![](http://latex.codecogs.com/gif.latex?A2%20%3D%20%280.1333-0.0666%29%5Ctimes0.6666%3D%5Cmathbf%7B0.04446222%7D)  
![](http://latex.codecogs.com/gif.latex?A3%20%3D%20%280.4-0.1333%29%5Ctimes0.4285%20%3D%5Cmathbf%7B0.11428095%7D)  
![](http://latex.codecogs.com/gif.latex?A4%20%3D%20%280.4666-0.4%29%5Ctimes0.3043%20%3D%5Cmathbf%7B0.02026638%7D)  
   
![](http://latex.codecogs.com/gif.latex?AP%20%3D%200.0666&plus;0.04446222&plus;0.11428095&plus;0.02026638)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%200.24560955)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cmathbf%7B24.56%5C%25%7D)  


<!--In order to evaluate your detections, you just need a simple list of `Detection` objects. A `Detection` object is a very simple class containing the class id, class probability and bounding boxes coordinates of the detected objects. This same structure is used for the groundtruth detections.-->

**Note again :** For the PASCAL VOC challenge, a prediction is positive if IoU > 0.5. However, if multiple detections of the same object are detected, it counts the first one as a positive while the rest as negatives

##### Summary
The key here is to compute the AP for each class, **in general** for computing Precision (P) and Recall (R) you must define what are: True Positives (TP), False Positives (FP), True Negative (TN) and False Negative (FN). Note that these may be different for different challenges and datasets.     
In the setting of Object Detection of the **Pascal VOC Challenge** are the following:       
- TP: are the Bounding Boxes (BB) that the intersection over union (IoU) with the ground truth (GT) is above 0.5
- FP: BB that the IoU with GT is below 0.5 also the BB that have IoU with a GT that has already been detected.
- TN: there are not true negative, the image are expected to contain at least one object
- FN: those images were the method failed to produce a BB           


### mAP (Mean Average Precision)
**A generic definition:** In order to calculate Mean Average Precision or mAP score, we take the _mean AP over all classes_ **and/or** over all IoU thresholds, depending on the competition
- VOC - PASCAL VOC2007 challenge only 1 IoU threshold of 0.5 was considered. So the mAP was averaged over all 20 object classes.
- COCO - For the COCO 2017 challenge, the mAP was calculated by averaging the AP over all 80 object categories **AND** all 10 IoU thresholds i.e AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05. So, First the AP is calculated for IoU threshold of 0.5 for each class i.e. We calculate the precision at every recall value(0 to 1 with a step size of 0.01), then it is repeated for IoU thresholds of 0.55,0.60,...,.95 and finally **average is taken over all the 80 classes and all the 10 thresholds** to get the metric used in the challenge.
