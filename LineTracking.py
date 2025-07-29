import cv2
import numpy as np
from djitellopy import Tello
import threading
import time


class LineFollower:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()

        # Control parameters
        self.is_running = False
        self.control_thread = None
        self.lock = threading.Lock()
        self.rc_control = [0, 0, 0, 0]  # lr, fb, ud, yaw

        # Image processing parameters
        self.scale_factor = 0.5
        self.kernel_size = 3

    def start(self):
        self.tello.takeoff()
        self.is_running = True
        self.control_thread = threading.Thread(target=self._send_control)
        self.control_thread.start()
        self._process_frames()

    def stop(self):
        self.is_running = False
        if self.control_thread:
            self.control_thread.join()
        self.tello.land()
        self.tello.streamoff()
        self.tello.end()  # 修复：使用 end() 而不是 disconnect()

    def _send_control(self):
        while self.is_running:
            with self.lock:
                self.tello.send_rc_control(*self.rc_control)
            time.sleep(0.05)

    def _process_frames(self):
        while self.is_running:
            frame = self.frame_read.frame

            if frame is None:
                continue

            # 1. Downsample image
            small_frame = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor)

            # 2. Convert to grayscale
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # 3. Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 4. Morphological closing (dilation followed by erosion)
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 5. Get last row for path detection
            last_row = closed[-1, :]

            # 6. Calculate path offset and control value
            control_value = self._calculate_control(last_row)

            # 7. Update control values
            with self.lock:
                self.rc_control = [control_value, 0, 0, 0]

            # Display processing results
            self._display_results(frame, small_frame, binary, closed)

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def _calculate_control(self, data):
        """Calculate control value from path data"""
        # Find indices of black pixels (path)
        black_pixels = np.where(data == 0)[0]

        if len(black_pixels) == 0:
            return 0  # No path detected

        # Calculate path center
        path_center = black_pixels[len(black_pixels) // 2]

        # Calculate image center
        image_center = len(data) // 2

        # Calculate offset
        offset = path_center - image_center

        # Scale control value
        control_value = offset // 3

        # Clamp to valid range (-100 to 100)
        return max(-100, min(100, control_value))

    def _display_results(self, original, small, binary, closed):
        """Display processing results for visualization"""
        # Resize images for display
        display_size = (320, 240)
        original_disp = cv2.resize(original, display_size)
        small_disp = cv2.resize(small, display_size)
        binary_disp = cv2.resize(binary, display_size)
        closed_disp = cv2.resize(closed, display_size)

        # Create combined display
        top_row = np.hstack((original_disp, cv2.cvtColor(small_disp, cv2.COLOR_GRAY2BGR)))
        bottom_row = np.hstack((cv2.cvtColor(binary_disp, cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor(closed_disp, cv2.COLOR_GRAY2BGR)))
        combined = np.vstack((top_row, bottom_row))

        # Add labels
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "Scaled", (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "Binary", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "Morphology", (330, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Line Following Processing', combined)


if __name__ == "__main__":
    follower = LineFollower()
    try:
        follower.start()
    except KeyboardInterrupt:
        follower.stop()
    finally:
        follower.stop()
