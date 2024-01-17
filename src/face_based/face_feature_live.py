import cv2

from src.face_based.FaceSample import FaceDetector


def main():
    print("Starting camera ... (this can take a while, I don't know why)")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_detector = FaceDetector(cap)

    face_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    while True:
        face_detector.update()

        if not face_detector.valid_faces_found():
            continue
        
        annotated_img = face_detector.last_img
        
        for fi, face in enumerate(face_detector.last_faces):
            tl_xy = round(face.tl_rxy * face_detector.video_dims)
            fwh = face.im.shape[:2]
            fxy = round(tl_xy + face.rxy * fwh)

            for i in range(len(face.features_rx)):
                x, y = round(tl_xy + face.features_rxy[i] * fwh)
                annotated_img = cv2.circle(annotated_img, (x,y), 3, (0,255,0), thickness=-1)
                annotated_img = cv2.putText(annotated_img, str(i), (x-10,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            
            annotated_img = cv2.circle(annotated_img, fxy, 3, (0,0,255), thickness=-1)
            # Bounding box
            annotated_img = cv2.rectangle(annotated_img, tl_xy, tl_xy + fwh, face_colors[fi % len(face_colors)], 3)

        cv2.imshow("webcam_annotated", annotated_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()

if __name__ == "__main__":
    main()