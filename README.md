# Image-Segmentation

## Image segmentation tool
This tool helps you to extract foreground and background from images.

![alt text](https://github.com/ASDAFI/Image-Segmentation/blob/master/assets/input.jpg)  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
![alt text](https://github.com/ASDAFI/Image-Segmentation/blob/master/assets/output.jpg)





## Prerequisites
run following command to install prerequisites.
``` bash
pip install -r requirements.txt
```


## GUI Guide
### Execution
Run program for your photo by following comand. (use your photo instead of `images/2.jpg` !)
```bash
python main.py -photo images/2.jpg
```
### Select foreground
you can select foreground by left click on the photo and after that, selected point will be shown by a green circle.

by doing left click again on each point, undo it.


### Select background
you can select background by right click on the photo and after that, selected point will be shown by a red circle.

by doing right click again on each point, undo it.


### Clear Screen
using `d` in your keyboard, clear all your screen from all selected points.

### Segmentation
you can do your segmentation proccess by pressing `c` on your keyborad.

### Save
you can save your file using `s` on your keyboard.

### ReSelect
you can reselect foreground and background by pressing `r` in keyboard.

## Developers Guide
following code shows you how to use this tool as a developer.
```python3
# import graph cut tools
import graph

# load image by graph
network = graph.Image(image)

# do cut by using some of foreground and background pixels
network.do_cut(some_foreground_pixels, some_background_pixels)

# extracting foreground and background pixels
foreground = network.get_foreground()
background = network.get_background()
```
