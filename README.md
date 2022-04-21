# 2021Spring_VfX_ImageStitching

## Usage
put all the images and two text files in one folder ([sample](data)) <br>
required text files:
- imlist.txt: image file name
- focal_length.csv: focal length of each image

### Feature Detection
```
python3 feature_detection.py [directory]
```
### Matching
```
python3 matching.py [directory]
```
### Stitching
```
python3 stitching.py [directory]
```

## Result
[original images](data) <br>
[result](data/result.png)
