# Install packages
```
pip3 -r install requirements.txt
```
If it doesn't works, try with Anaconda : https://www.anaconda.com/products/individual
# Try the face mask detection on a picture
```
python face_detector --image ./Examples/<image>
```

# Try the face mask detection on a video or a camera
If no arguments are provided, the detection is done on the camera
```
python face_detector_video [--video./Examples/<video>] 
```

# Start the Flask server
## On Windows
In the flask_serv folder
```
set FLASK_APP=main.py
flask run
```
## On Linux
```
export FLASK_APP=main.py
flask run
```
The trained model has been saved in the face_mask.model file.

## Acknowledgements
* Pyimagesearch.com
* Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems book
