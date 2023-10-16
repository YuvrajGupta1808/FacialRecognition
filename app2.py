import os
from PIL import Image
import matplotlib.pyplot as plt
from fer import FER

# Uncomment this if you're sure about the ffmpeg path
os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/local/bin/ffmpeg'
print(os.getcwd())

# Function to resize image
def resize_image(image_path, output_size=(640, 480)):
    img = Image.open(image_path)
    img = img.resize(output_size, Image.LANCZOS)
    img.save(image_path)

# Resize the image
resize_image("yuvraj.jpg")

# Read the resized image
img_senti = plt.imread("Virat/sad/8.jpeg")

# Initialize FER
emo_detector = FER(mtcnn=True)

# Detect emotions
captured_emotions = emo_detector.detect_emotions(img_senti)

# Display results
print(captured_emotions)
plt.imshow(img_senti)
plt.show()