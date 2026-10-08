import threading
import queue
import cv2
import numpy as np
from typing import Optional


class LiveWindow:
    """
    Displays a live video stream in an OpenCV window running on a background thread.
    """
    def __init__(
        self,
        window_title: str = "Live Window"
    ) -> None:
        """
        Args:
            window_title: Title for the OpenCV window.
        """
        self.window_title: str = window_title
        # Buffer queue that holds at most one unrendered frame
        self.frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)
        # Event to signal the display loop should stop
        self.stop_event: threading.Event = threading.Event()
        # Background thread running the display loop
        self.thread: threading.Thread = threading.Thread(
            target=self._display_loop,
            daemon=True
        )
        self.thread.start()

    def enqueue_frame(self, frame: np.ndarray):
        """
        Submit a new frame for display. If the queue is full, the oldest frame is dropped.

        Args:
            frame: Frame in BGR uint8 format. Shape:  (H, W, 3).
        """
        if self.stop_event.is_set():
            return
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop the stale frame and enqueue the latest one
            _ = self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(frame)

    def close(self):
        """
        Signal the display thread to exit and destroy the OpenCV window.
        """
        self.stop_event.set()
        # Unblock the queue.get() call with a sentinel
        self.frame_queue.put(None)
        self.thread.join()

    def _display_loop(self):
        """
        Internal method running in the daemon thread. Pulls frames from the queue
        and renders them in the OpenCV window until a stop signal is received.
        """
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)

        while not self.stop_event.is_set():
            frame = self.frame_queue.get()
            if frame is None:
                break

            # Skip rendering when the window is hidden or minimized
            visible = cv2.getWindowProperty(
                self.window_title,
                cv2.WND_PROP_VISIBLE
            )
            if visible < 1:
                cv2.waitKey(50)
                continue

            cv2.imshow(self.window_title, frame)
            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyWindow(self.window_title)