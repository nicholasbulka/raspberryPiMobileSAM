





so the masks are there, they are just appearing and disappearing too quickly to notice.

fix this soon.





Full details of frames:
Current frame: shape=(37, 30, 3), dtype=uint8, min=3, max=254
Previous frame: shape=(54, 96, 3), dtype=uint8, min=0, max=255
Traceback (most recent call last):
  File "/home/bw1/raspberryPiMobileSAM/pi5/utils.py", line 173, in get_change_bbox

since the masks are objects in 2d, we could implement a crude physics dx that relates to the motion that was detected colliding with the mask. Then send the mask off the screen and "release" the memory when it's fully left the bounds of the screen.