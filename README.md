11/16/24 - 

Raspberry pi 4 model B.
unicam camera for raspberry pi
Ubuntu 24.4
raspi-config install
ros2-jazzy
a docker installation of ros2-jazzy

python3 -m venv nameOfPreferredVenv

follow instructions to build mobile sam.

https://github.com/ChaoningZhang/MobileSAM



for raspberry pi 5, new camera - had to symlink system python libraries for the camera, everything for picamera2
for example

ln -s /system/path



   92  python3 -m venv mobileSamEnv
   93  source mobileSamEnv/bin/activate
   94  ln -s /usr/lib/python3/dist-packages/libcamera* samvenv/lib/python3.11/site-packages/
   95  ln -s /usr/lib/python3/dist-packages/libcamera* mobileSamEnv/lib/python3.11/site-packages/
   96  pip install -r requirements.txt 
   97  python main_rpi_5.py
   98  sudo apt-get install -y python3-libcamera python3-picamera2
   99  ln -s /usr/lib/python3/dist-packages/picamera1* mobileSamEnv/lib/python3.11/site-packages/
  100  python main_rpi_5.py
  101  ln -s /usr/lib/python3/dist-packages/picamera2* mobileSamEnv/lib/python3.11/site-packages/
  102  python main_rpi_5.py
  103  sudo apt-get install -y python3-v4l2
  104  ln -s /usr/lib/python3/dist-packages/v4l2* mobileSamEnv/lib/python3.11/site-packages/
  105  python main_rpi_5.py
  106  ln -s /usr/lib/python3/dist-packages/av* mobileSamEnv/lib/python3.11/site-packages/
  107  python main_rpi_5.py
  108  ln -s /usr/lib/python3/dist-packages/prctl* mobileSamEnv/lib/python3.11/site-packages/
  109  python main_rpi_5.py
  110  ln -s /usr/lib/python3/dist-packages/_prctl* mobileSamEnv/lib/python3.11/site-packages/
  111  python main_rpi_5.py
  112  ln -s /usr/lib/python3/dist-packages/piexif* mobileSamEnv/lib/python3.11/site-packages/
  113  python main_rpi_5.py
  114  ln -s /usr/lib/python3/dist-packages/pidng* mobileSamEnv/lib/python3.11/site-packages/
  115  python main_rpi_5.py
  116  ln -s /usr/lib/python3/dist-packages/simpljpeg* mobileSamEnv/lib/python3.11/site-packages/
  117  python main_rpi_5.py
  118  ln -s /usr/lib/python3/dist-packages/simplejpeg* mobileSamEnv/lib/python3.11/site-packages/
  119  python main_rpi_5.py
  120  which pip
  121  which pip3
  122  deactivate
  123  which pip
  124  which pip3
  125  ls -la /usr/bin/pip
  126  ls -la /usr/bin/pip3
  127  python3 -c "import numpy; print(f'Version: {numpy.__version__}'); print(f'Location: {numpy.__file__}')"
  128  source mobileSamEnv/bin/activate
  129  python -c "import numpy; print(f'Version: {numpy.__version__}'); print(f'Location: {numpy.__file__}')"
  130  pip uninstall numpy
  131  ln -s /usr/lib/python3/dist-packages/numpy* mobileSamEnv/lib/python3.11/site-packages/
  132  python main_rpi_5.py
  133  ln -s /usr/lib/python3/dist-packages/pykms* mobileSamEnv/lib/python3.11/site-packages/
