# Project-3
Hướng dẫn chạy thuật toán tracking:
Tải file yolov7-w6-pose.onnx trong drive
Để chạy thuật toán tracking đếm người cho video nào, có thể thay đường dẫn video ở file people_counting.py.
Để chạy với giải thuật SORT kết hợp với pose, sửa dòng thứ 356 của file sort4.py, có thể thay đổi trọng số của các ma trận iou, cos để so sánh pose bằng cosine similarity, oks để so sánh pose bằng OKS, và kpt để so sánh pose bằng Euclid
