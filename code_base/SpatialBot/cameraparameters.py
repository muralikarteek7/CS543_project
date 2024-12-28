from nuscenes.nuscenes import NuScenes


nusc = NuScenes(version='v1.0-mini', dataroot=r'C:\Users\Rohit\OneDrive\Desktop\cs543proj\dataset', verbose=True)

# fpr a sample i took the first image
sample = nusc.sample[0]

camera_token = sample['data']['CAM_FRONT']
camera_data = nusc.get('sample_data', camera_token)

calibrated_sensor_token = camera_data['calibrated_sensor_token']
calibrated_sensor = nusc.get('calibrated_sensor', calibrated_sensor_token)


intrinsics = calibrated_sensor['camera_intrinsic']


translation = calibrated_sensor['translation']  # Its in x, y, z(meters)
rotation = calibrated_sensor['rotation']        # Quaternion (w, x, y, z)
print("----------------iNTRINSIC MATRIX----------------")
for row in intrinsics:
    print(row)

print("\n-------------EXTRINSIC MATRIX-----------")
print(f"Translation : {translation}")
print(f"Rotation : {rotation}")
