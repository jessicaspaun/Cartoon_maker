# Cartoon_maker
Create a computer vision based style transfer for cartoonize animals

Tasks to do - start with dog
1. Get literature review of cartoonize
1. Create Face detector for dog 
  a. Find Eyes, ears, nose, tongue, mouth
2. Style transfer each piece of face


## face features
The mouth can be accessed through points [48, 68].
The right eyebrow through points [17, 22].
The left eyebrow through points [22, 27].
The right eye using [36, 42].
The left eye with [42, 48].
The nose using [27, 35].
And the jaw via [0, 17].

Based some code off of this tutorial | https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
You will need to use this repo for some functions | https://github.com/jrosebr1/imutils
