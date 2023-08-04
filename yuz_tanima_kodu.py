import face_recognition
import cv2
import numpy as np


video_capture = cv2.VideoCapture(0)


isim_image = face_recognition.load_image_file("isim.jpg")
isim_face_encoding = face_recognition.face_encodings(isim_image)[0]


isim_image = face_recognition.load_image_file("isim.jpg")
isim_face_encoding = face_recognition.face_encodings(isim_image)[0]

isim_image = face_recognition.load_image_file("isim.jpg")
isim_face_encoding = face_recognition.face_encodings(isim_image)[0]


isim_image = face_recognition.load_image_file("isim.jpg")
isim_face_encoding = face_recognition.face_encodings(isim_image)[0]

isim_image = face_recognition.load_image_file("isim.jpg")
isim_face_encoding = face_recognition.face_encodings(isim_image)[0]


isim_image = face_recognition.load_image_file("isim.jpg")
isim_face_encoding = face_recognition.face_encodings(isim_image)[0]

known_face_encodings = [
    isim_face_encoding,
    isim_face_encoding,
    isim_face_encoding,
    isim_face_encoding,
    isim_face_encoding,
    isim_face_encoding
]
known_face_names = [
    "isim",
    "isim",
    "isim",
    "isim",
    "isim",
    "isim"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Tek bir video karesi yakalayın
    ret, frame = video_capture.read()

    
    # Zaman kazanmak için videonun yalnızca diğer karelerini işleyin
    if process_this_frame:

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

      
        rgb_small_frame = small_frame[:, :, ::-1]
        
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Yüzün bilinen yüzlerle eşleşip eşleşmediğine bakın
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"


            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Sonucları goster
    for (top, right, bottom, left), name in zip(face_locations, face_names):
     #Tespit ettigimiz cerceve 1/4 boyuta olceklendiginden, yedek yuz konumlarini ölceklendirin
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Yuzun etrafina kare cizmek icin
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Yuzun altina bir ad içeren bir etiket cizer
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Cikmak icin "q" tusuna basınız
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı yeniler
video_capture.release()
cv2.destroyAllWindows()