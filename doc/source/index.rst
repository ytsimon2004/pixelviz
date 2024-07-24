Welcome to PixViz's documentation!
====================================


API Reference
-----------------

.. toctree::
   :maxdepth: 1

   api/pixviz.rst


Usage
----------

1. Load your video by clicking ``Load Video``
2. Drag your ROI(s) by first click ``Drag a Rect ROI`` button then drag area in video view. Press ``play`` see preview in plot view (delete by first click the roi in table, then click ``Delete ROI``), ``clear`` button is used for delete realtime plot display
3. Click ``Process`` the evaluate all, output will be saved as json meta and npy array in the same directory of video source
4. The data (.npy) can be reload using ``load result`` (the meta json need to be in the same directory)



To be implemented
------------------

Features
^^^^^^^^^

.. task-list::
   :custom:

   - [ ] Different shape of ROIs, allow upload reference figure, for zoom in and out
   - [ ] Save as .mat option


Enhancement
^^^^^^^^^^^^

.. task-list::
   :custom:

   - [ ] Resize issue after load video



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`