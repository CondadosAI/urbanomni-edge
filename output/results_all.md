| Size | Input | Hardware | Runtime | Model ms (median) | min-max over sessions | Sessions | Corner err. median / p90 (px) | Dets matched |
|---|--:|---|---|--:|--:|--:|--:|--:|
| N | 640 | RTX 3060 Laptop | PyTorch FP32 | 4.2 | 3.9-4.2 | 2 | reference | reference |
| N | 640 | RTX 3060 Laptop | TensorRT FP16 | 1.4 | 1.4-1.4 | 2 | 0.19 / 0.37 | 0.998 |
| N | 640 | Iris Xe iGPU | OpenVINO FP16 | 15.2 | 14.7-15.2 | 2 | 0.23 / 0.53 | 0.995 |
| N | 640 | Iris Xe iGPU | OpenVINO INT8 | 13.0 | 10.9-13.0 | 2 | 1.08 / 3.16 | 0.959 |
| N | 640 | Iris Xe iGPU | OpenVINO INT8, float head | 14.2 | 10.8-14.2 | 2 | 0.83 / 2.69 | 0.968 |
| N | 640 | i7-12700H CPU | OpenVINO FP32 | 18.2 | 16.6-18.2 | 2 | 0.02 / 0.06 | 0.998 |
| N | 640 | i7-12700H CPU | OpenVINO INT8 | 9.0 | 8.4-9.0 | 2 | 1.06 / 3.11 | 0.960 |
| N | 640 | i7-12700H CPU | OpenVINO INT8, float head | 10.6 | 10.0-10.6 | 2 | 0.78 / 2.63 | 0.975 |
| S | 640 | RTX 3060 Laptop | PyTorch FP32 | 6.7 | 6.7-6.7 | 2 | reference | reference |
| S | 640 | RTX 3060 Laptop | TensorRT FP16 | 2.0 | 2.0-2.0 | 2 | 0.19 / 0.40 | 0.998 |
| S | 640 | Iris Xe iGPU | OpenVINO FP16 | 30.8 | 28.6-30.8 | 2 | 0.24 / 0.62 | 0.995 |
| S | 640 | Iris Xe iGPU | OpenVINO INT8 | 22.5 | 20.5-22.5 | 2 | 0.96 / 2.76 | 0.970 |
| S | 640 | Iris Xe iGPU | OpenVINO INT8, float head | 16.7 | 14.4-25.7 | 3 | 0.50 / 2.14 | 0.985 |
| S | 640 | i7-12700H CPU | OpenVINO FP32 | 66.1 | 51.9-66.1 | 2 | 0.01 / 0.06 | 0.999 |
| S | 640 | i7-12700H CPU | OpenVINO INT8 | 20.8 | 18.9-20.8 | 2 | 0.96 / 2.68 | 0.970 |
| S | 640 | i7-12700H CPU | OpenVINO INT8, float head | 25.2 | 23.2-25.2 | 2 | 0.44 / 2.03 | 0.986 |
| M | 640 | RTX 3060 Laptop | PyTorch FP32 | 15.4 | 15.3-15.7 | 3 | reference | reference |
| M | 640 | RTX 3060 Laptop | TensorRT FP16 | 4.0 | 3.9-4.0 | 3 | 0.18 / 0.37 | 0.997 |
| M | 640 | Iris Xe iGPU | OpenVINO FP16 | 57.8 | 55.8-68.5 | 3 | 0.22 / 0.56 | 0.995 |
| M | 640 | Iris Xe iGPU | OpenVINO INT8 | 42.1 | 31.3-42.1 | 2 | 0.96 / 3.04 | 0.982 |
| M | 640 | Iris Xe iGPU | OpenVINO INT8, float head | 33.2 | 26.1-47.1 | 3 | 0.39 / 0.98 | 0.989 |
| M | 640 | i7-12700H CPU | OpenVINO FP32 | 160.9 | 156.1-165.8 | 3 | 0.01 / 0.02 | 0.999 |
| M | 640 | i7-12700H CPU | OpenVINO INT8 | 48.2 | 46.2-53.6 | 3 | 0.94 / 3.05 | 0.983 |
| M | 640 | i7-12700H CPU | OpenVINO INT8, float head | 62.7 | 59.6-69.8 | 3 | 0.33 / 0.90 | 0.988 |
| L | 640 | RTX 3060 Laptop | PyTorch FP32 | 19.8 | 19.4-19.8 | 2 | reference | reference |
| L | 640 | RTX 3060 Laptop | TensorRT FP16 | 5.1 | 5.1-5.1 | 2 | 0.18 / 0.36 | 0.998 |
| L | 640 | Iris Xe iGPU | OpenVINO FP16 | 88.9 | 73.4-88.9 | 2 | 0.21 / 0.53 | 0.995 |
| L | 640 | i7-12700H CPU | OpenVINO FP32 | 215.3 | 200.3-215.3 | 2 | 0.01 / 0.02 | 0.999 |
| X | 640 | RTX 3060 Laptop | PyTorch FP32 | 37.6 | 36.9-37.6 | 2 | reference | reference |
| X | 640 | RTX 3060 Laptop | TensorRT FP16 | 10.1 | 9.9-10.1 | 2 | 0.18 / 0.36 | 0.999 |
| X | 640 | Iris Xe iGPU | OpenVINO FP16 | 171.5 | 137.6-171.5 | 2 | 0.24 / 0.73 | 0.997 |
| X | 640 | i7-12700H CPU | OpenVINO FP32 | 463.6 | 447.2-463.6 | 2 | 0.01 / 0.02 | 1.000 |
| N | 960 | RTX 3060 Laptop | PyTorch FP32 | 7.0 | 7.0-7.0 | 2 | reference | reference |
| N | 960 | RTX 3060 Laptop | TensorRT FP16 | 2.3 | 2.3-2.3 | 2 | 0.20 / 0.37 | 0.997 |
| N | 960 | Iris Xe iGPU | OpenVINO FP16 | 29.8 | 28.9-29.8 | 2 | 0.22 / 0.45 | 0.995 |
| N | 960 | Iris Xe iGPU | OpenVINO INT8 | 28.9 | 20.9-28.9 | 2 | 1.38 / 3.10 | 0.919 |
| N | 960 | Iris Xe iGPU | OpenVINO INT8, float head | 29.6 | 20.0-29.6 | 2 | 0.82 / 2.37 | 0.944 |
| N | 960 | i7-12700H CPU | OpenVINO FP32 | 38.1 | 36.5-38.1 | 2 | 0.01 / 0.05 | 0.999 |
| N | 960 | i7-12700H CPU | OpenVINO INT8 | 37.8 | 17.3-37.8 | 2 | 1.35 / 3.03 | 0.921 |
| N | 960 | i7-12700H CPU | OpenVINO INT8, float head | 29.8 | 21.4-29.8 | 2 | 0.78 / 2.30 | 0.952 |
| S | 960 | RTX 3060 Laptop | PyTorch FP32 | 14.9 | 14.5-14.9 | 2 | reference | reference |
| S | 960 | RTX 3060 Laptop | TensorRT FP16 | 4.1 | 4.1-4.1 | 2 | 0.19 / 0.34 | 0.995 |
| S | 960 | Iris Xe iGPU | OpenVINO FP16 | 79.9 | 55.6-79.9 | 2 | 0.22 / 0.53 | 0.994 |
| S | 960 | Iris Xe iGPU | OpenVINO INT8 | 45.0 | 38.7-45.0 | 2 | 1.49 / 4.42 | 0.940 |
| S | 960 | Iris Xe iGPU | OpenVINO INT8, float head | 46.8 | 34.7-46.8 | 2 | 0.75 / 3.40 | 0.962 |
| S | 960 | i7-12700H CPU | OpenVINO FP32 | 127.2 | 116.5-127.2 | 2 | 0.01 / 0.03 | 0.998 |
| S | 960 | i7-12700H CPU | OpenVINO INT8 | 47.4 | 43.6-47.4 | 2 | 1.46 / 4.45 | 0.942 |
| S | 960 | i7-12700H CPU | OpenVINO INT8, float head | 58.6 | 55.9-58.6 | 2 | 0.69 / 3.39 | 0.961 |
| M | 960 | RTX 3060 Laptop | PyTorch FP32 | 33.9 | 33.2-33.9 | 2 | reference | reference |
| M | 960 | RTX 3060 Laptop | TensorRT FP16 | 8.3 | 8.2-8.3 | 2 | 0.19 / 0.32 | 0.999 |
| M | 960 | Iris Xe iGPU | OpenVINO FP16 | 181.6 | 121.1-181.6 | 2 | 0.22 / 0.50 | 0.995 |
| M | 960 | Iris Xe iGPU | OpenVINO INT8 | 95.2 | 77.0-95.2 | 2 | 2.12 / 4.61 | 0.971 |
| M | 960 | Iris Xe iGPU | OpenVINO INT8, float head | 103.1 | 75.5-103.1 | 2 | 0.61 / 2.23 | 0.983 |
| M | 960 | i7-12700H CPU | OpenVINO FP32 | 386.5 | 357.4-386.5 | 2 | 0.01 / 0.02 | 1.000 |
| M | 960 | i7-12700H CPU | OpenVINO INT8 | 157.9 | 114.3-157.9 | 2 | 2.13 / 4.60 | 0.970 |
| M | 960 | i7-12700H CPU | OpenVINO INT8, float head | 155.5 | 142.7-155.5 | 2 | 0.60 / 2.27 | 0.981 |
| L | 960 | RTX 3060 Laptop | PyTorch FP32 | 43.1 | 42.2-43.1 | 2 | reference | reference |
| L | 960 | RTX 3060 Laptop | TensorRT FP16 | 10.9 | 10.7-10.9 | 2 | 0.19 / 0.33 | 0.998 |
| L | 960 | Iris Xe iGPU | OpenVINO FP16 | 236.3 | 157.8-236.3 | 2 | 0.22 / 0.52 | 0.997 |
| L | 960 | i7-12700H CPU | OpenVINO FP32 | 498.5 | 448.3-498.5 | 2 | 0.01 / 0.02 | 1.000 |
| X | 960 | RTX 3060 Laptop | PyTorch FP32 | 83.8 | 82.0-83.8 | 2 | reference | reference |
| X | 960 | RTX 3060 Laptop | TensorRT FP16 | 20.9 | 20.9-21.2 | 3 | 0.19 / 0.33 | 0.998 |
| X | 960 | Iris Xe iGPU | OpenVINO FP16 | 303.9 | 260.6-402.8 | 3 | 0.24 / 0.87 | 0.996 |
| X | 960 | i7-12700H CPU | OpenVINO FP32 | 1118.4 | 985.5-1118.4 | 2 | 0.01 / 0.02 | 0.999 |
