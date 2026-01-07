# ml_and_dl_stuff
this repo contains scripts in the context of machine learning and deep learning. The main motivation is to play around and learn about it.

## New Concept for Organizing Images

Example queries:
* "gib mir alle Bilder auf denen Janine und Joris drauf sind im Urlaub in Südfrankreich letztes Jahr."
* "gib mir alle Bilder wo wir am Strand baden waren."
* etc.

Main Points:
* LLM to interpret user query
* "hard" SQL filter on postgres database
* CLIP embedding search in FAISS
* return found images

Meta tags per image in postgres:
* date
* gps location
* camera name
* probability for a recognized face
  * prob_janine
  * prob_michi
  * prob_joris
  * prob_nele
* gefundene Objekte auf dem Bild
* thumbnails
* reference to original image (pcloud, local)
* marked as deleted? (usecase: ich möchte das Bild löschen)

Meta tags per image in FAISS:
* CLIP embedding

Pre-Reqs.:
* label detected faces
* train classifier based on cropped face embeddings.

Docker-Compose Setup:
* Python mit CUDA + Pytorch etc. + Modelle (CLIP, ArcNet, Retina, LLama)
* Postgres + Mount auf DB
* FAISS + Mount auf DB
* Volume Mount auf die Bilder
* Front-End?! (Django? Flask? etc.)

Workflow:
* Sync with pcloud to local
* extraction scripts
  * File Name, File Path
  * EXIF Header: date, gps, camera
  * Create thumbnail
  * face recon: prob_janine, etc.
  * scene: CLIP embedding

Other Reqs.:
* nicht benötigte Bilder löschen (--> vermutl in pcloud)


## bazel/dazel
for the sake of getting to know bazel and dazel, this repo is using it. \

for example:
```
bazel run image_tagger:main -- -list_classes
```

https://docs.bazel.build/versions/main/be/python.html \
https://github.com/nadirizr/dazel \

## image tagger
This script aims at running pre-trained models (e.g. yolo5) on images and to infer detections that are stored in a sqlite database. \
The cmd line tool also allows to query for certain classes and retrieve images that contain the classes at a given confidence level. \
It is based on pytorch. \

TODO: Plan is to extend this by face detection and to train an own face recognition model based on own training data!

## alexnet
https://learnopencv.com/pytorch-for-beginners-basics/

https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/


