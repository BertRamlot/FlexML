import cv2

from src.FaceDetector import FaceDetector


def main():
    print("Starting camera")
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector(cap)

    face_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    while True:
        face_detector.update()
        
        annotated_img = face_detector.last_img
        
        for fi, face in enumerate(face_detector.last_faces):
            tl_x = round(face.tl_rx * face_detector.video_width)
            tl_y = round(face.tl_ry * face_detector.video_height)
            fw = face.im.shape[1]
            fh = face.im.shape[0]
            fx = round(tl_x + face.rx * fw)
            fy = round(tl_y + face.ry * fh)

            for i in range(len(face.features_rx)):
                x = round(tl_x + face.features_rx[i] * fw)
                y = round(tl_y + face.features_ry[i] * fh)
                annotated_img = cv2.circle(annotated_img, (x,y), 3, (0,255,0), thickness=-1)
                annotated_img = cv2.putText(annotated_img, str(i), (x-10,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            
            annotated_img = cv2.circle(annotated_img, (fx, fy), 3, (0,0,255), thickness=-1)

            annotated_img = cv2.rectangle(annotated_img, (tl_x, tl_y), (tl_x+fw, tl_y+fh), face_colors[fi % len(face_colors)], 3)


        cv2.imshow("webcam_annotated", annotated_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    main()