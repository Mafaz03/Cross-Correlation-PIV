# Parallel Cross-Correlation PIV

This project implements a **parallelized Cross-Correlation algorithm** for Particle Image Velocimetry (PIV), enabling efficient computation of velocity fields from video sequences.

Two different parallel approaches are explored. For detailed methodology and theory, please refer to the accompanying PDF.

---

# Overview

- Extracts velocity components (**U, V**) from video frames  
- Supports both **serial** and **MPI-based parallel implementations**  
- Generates **flow visualizations** (contours + quiver plots)  
- Designed for scalability and performance experimentation  

---

# Workflow

## Step 1: (Optional) Trim Input Video

Reduce the number of frames for faster experimentation:

```
python3 crop_framelength.py \
    --video_path path/to/input.mp4 \
    --video_save_path path/to/output.mp4 \
    --frames <num_frames>
```

## Step 1: Compute Velocity Fields
### Serial Version

```
./cc_serial.out <video_path> <save_U_path> <save_V_path> <save_img_path> <skip>
```

#### Example:
```
./cc_serial.out Clips/cube_clip_benchmark.mp4 Data/Cube/Velocities/U.dat Data/Cube/Velocities/V.dat Data/Cube/Images/img.png 1
```

### Parallel Version 1 (MPI)

```
mpirun -np <num_processes> ./cc_p_v1.out \
    <video_path> <save_U_path> <save_V_path> <save_img_path> \
    <skip> <num_threads> <verbose>
```

#### Example:
```
mpirun -np 4 ./cc_p_v1.out Clips/cube_clip_benchmark.mp4 Data/Cube/Velocities/U.dat Data/Cube/Velocities/V.dat Data/Cube/Images/img.png 1 4 1
```

### Parallel Version 1 (Optimized MPI)

```
mpirun -np <num_processes> ./cc_p_v2.out \
    <video_path> <save_U_path> <save_V_path> <save_img_path> \
    <skip> <num_threads> <verbose>
```

#### Example:
```
mpirun -np 4 ./cc_p_v2.out Clips/cube_clip_benchmark.mp4 Data/Cube/Velocities/U.dat Data/Cube/Velocities/V.dat Data/Cube/Images/img.png 1 4 1
```

### Animate results

```
python3 animate.py --image_folder <image folder location> \
    --velocity_folder <velocity folder location> \
    --maximum_frames <maximum frames to animate> \
    --fps <fps> --save_path <save path location> \
    --pixel_to_meter <calibration> --dt <time between frames (s)>
```

#### Example
```
python3 animate.py --image_folder Data/Cube/Images --velocity_folder Data/Cube/Velocities --maximum_frames 12 --fps 20 --save_path Result_clips/cube.mp4 --pixel_to_meter 0.000198019802 --dt 0.002941176471
```

## Result

![Cube](assets/cube.png)
![Cylinder](assets/cylinder.png)
![Rods](assets/rods.png)
![Airfoil](assets/airfoil.png)