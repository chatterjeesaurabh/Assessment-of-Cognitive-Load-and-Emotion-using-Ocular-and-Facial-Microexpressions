To just run the model on any desired face video to get Facial Action Unit intensities:

* First use `video_face_detector_save_tool.py` in 'AU_Recognition/data' folder to extract faces from videos frame by frame and store them as images in a folder named as the video file name.
* A CSV file is also produced with the images save locations, with `data.csv` - which will be used by the model inference code to fetch the face images and predic the Face Action Unit Intensities.
* Then run the model using `inference.sh` or `inference.py`.
