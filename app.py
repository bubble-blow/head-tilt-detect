import math
import sys
from dataclasses import dataclass

import cv2
import mediapipe as mp
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263


@dataclass
class PostureResult:
    frame_rgb: any
    angle: float | None
    status_text: str
    ok: bool


class PoseMonitorThread(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, threshold: float = 6.0):
        super().__init__()
        self._threshold = threshold
        self._running = True

    def set_threshold(self, threshold: float) -> None:
        self._threshold = threshold

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.result_ready.emit(
                PostureResult(
                    frame_rgb=None,
                    angle=None,
                    status_text="无法打开摄像头，请检查权限或设备。",
                    ok=False,
                )
            )
            return

        mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        while self._running:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)

            angle = None
            status_text = "未检测到人脸"
            ok = False

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                h, w, _ = frame.shape

                left = face.landmark[LEFT_EYE_OUTER_IDX]
                right = face.landmark[RIGHT_EYE_OUTER_IDX]
                lx, ly = int(left.x * w), int(left.y * h)
                rx, ry = int(right.x * w), int(right.y * h)

                cv2.line(frame, (lx, ly), (rx, ry), (64, 196, 255), 2)
                cv2.circle(frame, (lx, ly), 4, (130, 255, 130), -1)
                cv2.circle(frame, (rx, ry), 4, (130, 255, 130), -1)

                angle = math.degrees(math.atan2(ry - ly, rx - lx))
                if abs(angle) <= self._threshold:
                    status_text = "坐姿良好：头部基本水平"
                    ok = True
                else:
                    direction = "左倾" if angle > 0 else "右倾"
                    status_text = f"请调整姿势：头部{direction}"

                cv2.putText(
                    frame,
                    f"Angle: {angle:.1f} deg",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (80, 230, 120) if ok else (80, 120, 230),
                    2,
                    cv2.LINE_AA,
                )

            out_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.result_ready.emit(
                PostureResult(
                    frame_rgb=out_rgb,
                    angle=angle,
                    status_text=status_text,
                    ok=ok,
                )
            )

        mesh.close()
        cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("坐姿检测助手（PyQt + MediaPipe）")
        self.resize(1040, 700)

        self.monitor_thread: PoseMonitorThread | None = None

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)

        self.video_label = QLabel("点击“开始检测”启动摄像头")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 540)
        self.video_label.setStyleSheet(
            """
            QLabel {
                background-color: #101826;
                border: 2px solid #2f4863;
                border-radius: 14px;
                color: #9eb6cc;
                font-size: 18px;
            }
            """
        )

        control_panel = QFrame()
        control_panel.setFixedWidth(280)
        control_panel.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #1a2b3f, stop:1 #122033);
                border-radius: 14px;
                border: 1px solid #33506f;
            }
            QLabel {
                color: #e1ecf7;
            }
            """
        )

        panel_layout = QVBoxLayout(control_panel)
        panel_layout.setContentsMargins(16, 18, 16, 18)
        panel_layout.setSpacing(16)

        title = QLabel("坐姿状态")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))

        self.status_label = QLabel("等待启动")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "color: #b8d6ff; background:#0f1d2d; border-radius:10px; padding:12px;"
        )

        self.angle_label = QLabel("眼线角度：--")
        self.threshold_label = QLabel("容忍阈值：6°")

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(2)
        self.threshold_slider.setMaximum(15)
        self.threshold_slider.setValue(6)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setEnabled(False)

        for btn in (self.start_btn, self.stop_btn):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(42)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #2d8cff;
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-size: 16px;
                    font-weight: 600;
                }
                QPushButton:disabled {
                    background-color: #516b8a;
                    color: #d4dde7;
                }
                QPushButton:hover:!disabled {
                    background-color: #56a0ff;
                }
                """
            )

        self.stop_btn.setStyleSheet(
            self.stop_btn.styleSheet()
            + "QPushButton { background-color: #ef5350; } QPushButton:hover:!disabled { background-color:#f27876; }"
        )

        self.start_btn.clicked.connect(self.start_monitor)
        self.stop_btn.clicked.connect(self.stop_monitor)

        panel_layout.addWidget(title)
        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(self.angle_label)
        panel_layout.addWidget(self.threshold_label)
        panel_layout.addWidget(self.threshold_slider)
        panel_layout.addStretch()
        panel_layout.addWidget(self.start_btn)
        panel_layout.addWidget(self.stop_btn)

        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addWidget(control_panel)

        self.setStyleSheet("QMainWindow { background-color: #0c1422; }")

    def on_threshold_changed(self, value: int) -> None:
        self.threshold_label.setText(f"容忍阈值：{value}°")
        if self.monitor_thread:
            self.monitor_thread.set_threshold(float(value))

    def start_monitor(self) -> None:
        if self.monitor_thread and self.monitor_thread.isRunning():
            return

        self.monitor_thread = PoseMonitorThread(threshold=float(self.threshold_slider.value()))
        self.monitor_thread.result_ready.connect(self.update_ui)
        self.monitor_thread.finished.connect(self.on_thread_finished)
        self.monitor_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("正在启动摄像头...")

    def stop_monitor(self) -> None:
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.monitor_thread.wait(1500)
        self.on_thread_finished()

    def on_thread_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_ui(self, result: PostureResult) -> None:
        if result.frame_rgb is not None:
            h, w, ch = result.frame_rgb.shape
            bytes_per_line = ch * w
            image = QImage(result.frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(image)
            pix = pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.video_label.setPixmap(pix)

        self.status_label.setText(result.status_text)
        self.status_label.setStyleSheet(
            (
                "color: #d3ffdf; background:#173624; border-radius:10px; padding:12px;"
                if result.ok
                else "color: #ffd6d6; background:#472126; border-radius:10px; padding:12px;"
            )
        )

        if result.angle is None:
            self.angle_label.setText("眼线角度：--")
        else:
            self.angle_label.setText(f"眼线角度：{result.angle:.2f}°")

    def closeEvent(self, event):
        self.stop_monitor()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
