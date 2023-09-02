#! /usr/bin/python

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from playsound import playsound
import face_recognition
import imutils
import pickle
import time
import cv2
import RPi.GPIO as GPIO
import warnings
import time
import telepot
from datetime import datetime

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
bot_token = '6018390048:AAEDXd4h1i11E4LqLiFdloMiYexOWwNGcko'

# Replace 'CHAT_ID' with the chat ID where you want to send the message
chat_id = '1409680610'

def send_message(message):
    bot = telepot.Bot(bot_token)
    bot.sendMessage(chat_id, message)

warnings.filterwarnings('ignore')

# Pin and sound definition
button_pin = 15
relay_pin = 17
sound_path = "/home/techno-media/facial_recognition/doorlock.mp3"
trigger = False

#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "/home/techno-media/facial_recognition/encodings.pickle"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# initialize the video stream and allow the camera sensor to warm up
# Set the ser to the followng
# src = 0 : for the build in single web cam, could be your laptop webcam
# src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
#vs = VideoStream().start()
vs = VideoStream().start()
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# Setup pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_pin, GPIO.OUT)
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize button state
button_state = GPIO.input(button_pin)

# start time
start_time = time.time()
selisih_time = time.time() - start_time

# Turn off relay_pin (kunci pintu)
GPIO.output(relay_pin, False)


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    # Detect the fce boxes
    boxes = face_recognition.face_locations(frame)
    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []
    
    # Read the state of the button
    new_button_state = GPIO.input(button_pin)
    print(new_button_state)
    
    if new_button_state != button_state:
        button_state = new_button_state

        if button_state == GPIO.LOW:  # Button is pressed (due to pull-up resistor)
            # Toggle the relay state
            GPIO.output(relay_pin, True)
            
            start_time = time.time()
            
    if((selisih_time) >= 8 and trigger == True):
        print('turn off')
        
        # Turn off relay_pin (kunci pintu)
        GPIO.output(relay_pin, False)
        
        # Turn off trigger
        trigger = False
        
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown" #if face is not recognized, then print Unknown

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            
            # start time
            print('turn on')
            # Turn on relay_pin (buka pintu)
            GPIO.output(relay_pin, True)
            start_time = time.time()
            
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select second entry in the dictionary)
            name = max(counts, key=counts.get)

            #If someone in your dataset is identified, print their name on the screen
            if currentname != name:
                currentname = name
                print(currentname)
                
            if trigger == False:
                playsound(sound_path)
                print('masuk')
                # SEND DATA TO TELEGRAM
                message = f"{name} masuk ke rumah pada pukul {datetime.now()}!"
                send_message(message)
                trigger = True
        
        # update the list of names
        names.append(name)  
    
    selisih_time = time.time() - start_time
    
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            .8, (0, 255, 255), 2)

    # display the image to our screen
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # quit when 'q' key is pressed
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()


# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()