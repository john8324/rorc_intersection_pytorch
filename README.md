# rorc_intersection_pytorch
Intersection of 2D Rotated Rectangle implemented by PyTorch

## How to Use
1. Install Python3, PyTorch, and CUDA (optional but recommended if possible)
2. Place rorc_intersection.py at the side of your code
3. `from rorc_intersection import rotatedRectangleIntersection, get_intersection_area`

## Format
`torch.zeros(N, 5, dtype=torch.float32)`

The columns represent CenterX, CenterY, Width, Height, and Angle (clockwise, degree), respectively.


## Ref 1:
https://github.com/opencv/opencv/blob/93d490213fd6520c2af9d2b23f0c99039d355760/modules/imgproc/src/intersection.cpp
## Ref 2:
https://github.com/lilanxiao/Rotated_IoU/blob/e2ca1530828ff64c105a53eb23e0788262d72428/box_intersection_2d.py





