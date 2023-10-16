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

# Initialize FER
emo_detector = FER(mtcnn=True)

virat_folder_path = "Virat"
processed_count = 0

image_files = [f for f in os.listdir(virat_folder_path) if f.endswith(('.jpeg', '.jpg', '.webp'))]

for image_file in image_files:
    image_path = os.path.join(virat_folder_path, image_file)
    resize_image(image_path)
    img_senti = plt.imread(image_path)
    
    # Detect emotions
    captured_emotions = emo_detector.detect_emotions(img_senti)

    if captured_emotions:
        max_emotion = max(captured_emotions[0]['emotions'], key=captured_emotions[0]['emotions'].get)
        emotion_folder = os.path.join(virat_folder_path, max_emotion)

        if not os.path.exists(emotion_folder):
            os.makedirs(emotion_folder)

        new_path = os.path.join(emotion_folder, image_file)
        os.rename(image_path, new_path)
        processed_count += 1

print(f"Processed {processed_count} images and categorized them based on emotions.")
