import cv2 as cv
import numpy as np
import os
import time
# Method which reads  the .flo files and converts into an numpy array.
start_cpu_time = time.process_time()
def readFlow(name):
    f = open(name, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

# Read the images from the directory
images_dir = "/Users/abhishekharikumarnarayanan/Desktop/MSc ACS Project /Optical Flow Estimator/middlebury/training/final/market_5"

# Read the Ground Truth from the directory
flow_dir = "/Users/abhishekharikumarnarayanan/Desktop/MSc ACS Project /Optical Flow Estimator/middlebury/training/flow/alley_1"
flow_files = sorted([file for file in os.listdir(flow_dir) if file.endswith(".flo")])
t_epe = 0
t_mse = 0
num_frames = 0
cv.namedWindow("Dense Optical Flow", cv.WINDOW_NORMAL)
for frame_filename in os.listdir(images_dir):
    frame_path = os.path.join(images_dir, frame_filename)
    frame = cv.imread(frame_path)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if num_frames > 0:
        # Calculate dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 7, 1.2, 0)

        ground_truth_flow = readFlow(os.path.join(flow_dir,flow_files[num_frames - 1]))

        calculated_flow_resized = cv.resize(flow, (1024, 436))

        diff = ground_truth_flow[..., :2] - calculated_flow_resized
        squared_difference_u = np.square(diff[..., 0])
        squared_difference_v = np.square(diff[..., 1])

        mse_u = np.mean(squared_difference_u)
        mse_v = np.mean(squared_difference_v)

        stu = ground_truth_flow[..., 0]
        stv = ground_truth_flow[..., 1]
        su = calculated_flow_resized[..., 0]
        sv = calculated_flow_resized[..., 1]

        epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
        mepe = np.mean(epe)
        mse = (mse_u + mse_v) / 2
        print(f"Frame {num_frames}: Endpoint Error: {mepe:.2f}, Mean Squared Error: {mse:.2f}")

        t_epe += mepe
        t_mse += mse
        cv.resizeWindow("Dense Optical Flow", 1024, 436)

        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # Opens a new window and displays the output frame
        cv.imshow("Dense Optical Flow", rgb)
        cv.waitKey(500)

    num_frames += 1
    prev_gray = gray

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# Calculating the Average Error metrics
aepe = t_epe / num_frames
amse = t_mse / num_frames

print("Average Endpoint Error:", aepe)
print("Average Mean Squared Error:", amse)
end_cpu_time = time.process_time()
elapsed_cpu_time = end_cpu_time - start_cpu_time
print(f"Elapsed CPU time: {elapsed_cpu_time:.4f} seconds")
cv.destroyAllWindows()


