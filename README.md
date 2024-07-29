# pixviz
![PyPI - Version](https://img.shields.io/pypi/v/pixviz)

## Pixel intensity analysis and visualization for image sequences

# Installation

- create conda env `conda create -n pixviz python~=3.10.0 -y`
- install dependencies `pip install pixviz`
- launch GUI `python -m pixviz`


# Simple Demo
![pixel_demo.gif](doc%2Fpixviz_demo.gif)


# GUI Usage

1. Load your video by clicking `Load Video`
2. Drag your ROI(s) by first click `Drag a Rect ROI` button then drag area in video view (left click inside the area for *moving*, left click the round red button for *rotation*)
3. Press `play` see preview in plot view (delete by first click the roi in table, then click ``Delete ROI``), ``clear`` button is used for delete realtime plot display
4. Click `Process` the evaluate all, output will be saved as json meta and npy array in the same directory of video source
5. The data (.npy) can be reload using `load result` (the meta json need to be in the same directory)


# API doc

- See [API Doc](https://pixelviz.readthedocs.io/en/latest/)