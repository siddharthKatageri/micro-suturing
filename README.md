# Micro Suturing Effectualness
### Problem Statement:
Design a vision-based techniques to evaluate effectualness of micro
suturing by trainee neurosurgeons.

### Parameters:
In total 6 parameters were decided, to evaluate a given suture.
This github repo focuses on 2 parameters to evaluate "knots" in suture. <br>
Namely,
1. Alignment of knots
2. Uniformity in distance between knots

### Objectives:
1. Detection of knots in suturing images.
2. Perform parametric evaluation for effectualness for micro-suturing.

### Preprocessing:
One of the major steps was preprocessing, as the suture images had noise and
illumination effect which needed to dealt with. Mode Clipping of images was done
which reduced noise, illumination effect and good segregation of background and
foreground. <br>
In order to reduce the search space for knots, segmentation and gaussian model
based noise reduction followed by contour hunting was done. Which reduced to search
space drastically.

### Knot Detection:
We wanted a intensity invariant classifier, to achieve this a SVM classifier was
trained on HOG feature descriptors of knots and non-knots patches.
In order to improve the performance, generative algorithms like GDA and GMM
were also implemented as a post processing step.

### Code:
1. To train a SVM classifier on HOG feature descriptors of knots and
non knots patches.
```
python svm_extra_features.py
```

2. To train GDA classifier.
```
python biclass_gda_with_save.py
```

3. To train GMM classifier.
```
python gmm_classifier.py
```

4. To generate prediction boxes on test set (SVM, NMS, GDA)
```
python svm+nms+gda_boxes_save.py
```

5. To generate prediction boxes on test set (SVM, NMS, GMM)
```
python generate_prediction_boxes.py
```
6. To test on given single test image
```
python main.py
```
7. Evaluated test images
```
python evaluation.py
```
