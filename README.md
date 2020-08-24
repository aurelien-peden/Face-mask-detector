# Description
Proof of concept made during an internship at Paris 12 Val de Marne University.
Developed a proof of concept web app using Flask to localize faces with OpenCV and detect if a person is wearing a face mask in a picture or a video. Used a dataset of +1000 images of people wearing face mask and reached an accuracy of 97% using Transfer Learning with MobileNet-v2 and TensorFlow 2.0/Keras. Project, report and slides used by teachers in classes for next year’s students.

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

# Project Report and Presentation Slides (In French)
* [Project Report](https://github.com/aurelien-peden/Face-mask-detector/blob/master/Rapport%20projet%20IA%20masque%20Aur%C3%A9lien%20Peden.pdf)
* [Presentation Slides](https://github.com/aurelien-peden/Face-mask-detector/blob/master/AI%20Port%20du%20masque%20slides.pptx)

# Acknowledgements
* Pyimagesearch.com
* Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems book
