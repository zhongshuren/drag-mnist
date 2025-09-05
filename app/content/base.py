import cv2


class ContentManager:
    def __init__(self):
        self.iterating = True

    def render(self):
        raise NotImplementedError

    def update_state(self, data):
        raise NotImplementedError

    def reset_state(self):
        raise NotImplementedError

    def stop(self):
        self.iterating = False

    def get_frame(self):
        self.iterating = True

        while self.iterating:
            frame = self.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
            frame = buffer.tobytes()
            # Concatenate frame and yield for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')