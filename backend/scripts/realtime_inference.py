import argparse
import time

import cv2
import requests


def main(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ok_enc, enc = cv2.imencode(".jpg", frame)
        if ok_enc:
            files = {"image": ("frame.jpg", enc.tobytes(), "image/jpeg")}
            resp = requests.post(f"{args.api}/emotion/frame", files=files, timeout=5)
            if resp.status_code == 200:
                payload = resp.json()
                label = payload["label"]
                conf = payload["confidence"]
                cv2.putText(frame, f"{label} ({conf:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Emotion Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    main(parser.parse_args())
