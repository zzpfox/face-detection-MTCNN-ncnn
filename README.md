# face-detection-MTCNN-ncnn
Implement MTCNN for face detection and landmark localization in ncnn framework in C++
Support Windows, Ubuntu and Raspberry Pi

Refactor from MTCNN v1 Matlab https://github.com/kpzhang93/MTCNN_face_detection_alignment

## Results
![image](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/examples.png)
![image](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/result.png)

## Require Dependency Library
  - OpenCV 3.3.1 (For displaying demo video)

## Compile in Windows 
```sh
$ ./build_vs2017_x64.bat
```

## Compile in Ubuntu 
```sh
$ sh build_ubuntu.sh
```

## Compile in Raspberry Pi 
```sh
$ sh build_raspberry.sh
```