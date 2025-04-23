#@title yolo
model = YOLO("yolov8n-seg.pt")

# 对输入图片进行检测（分割）
results = model(input_image_path)


img = cv2.imread(input_image_path)
height, width = img.shape[:2]

# 初始化字典，存储每个关键词的合并 mask（全零矩阵）
combined_masks = {k: np.zeros((height, width), dtype=np.uint8) for k in keywords}
label_detected = []
# 遍历检测结果，提取分割 mask
for result in results:
    if result.masks is not None:
        # 遍历每个检测结果
        for i in range(len(result.boxes)):
            # 获取类别索引和对应标签
            class_id = int(result.boxes.cls[i])
            label = model.names[class_id]
            if label in keywords:
                label_detected.append(label)
                # 获取当前检测对应的 mask（结果为 tensor）
                mask = result.masks.data[i].cpu().numpy()
                # 二值化 mask（阈值可调整）
                mask = (mask > 0.2).astype(np.uint8) * 255
                # 若 mask 尺寸与原图不同，则调整大小
                if mask.shape[0] != height or mask.shape[1] != width:
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))  # 10 pixels on each side ≈ 21x21 kernel 2*10 + 1
                mask = cv2.dilate(mask, kernel, iterations=1)
                # 合并同一类别的 mask（按位或操作）
                combined_masks[label] = cv2.bitwise_or(combined_masks[label], mask)

# 保存每个关键词对应的 mask 图片


maskFolder = "maskFolder"
# Create the folder if it doesn't exist
os.makedirs(maskFolder, exist_ok=True)

for label, mask in combined_masks.items():
    mask_path = f"mask_{label}.png"
    output_image_path = os.path.join(maskFolder, mask_path)
    cv2.imwrite(output_image_path, mask)
    print(f"Saved mask for '{label}' as {output_image_path}")