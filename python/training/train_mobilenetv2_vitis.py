import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ==========================================
# CẤU HÌNH PARAMETERS
# ==========================================
DATASET_DIR = "/home/iec/Parallel_Computing_on_FPGA/Vitis-AI/mobilenetv2_float/tf_mobilenetv2-1.0_3.5/data/combined/spectrograms"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3 # ['Healthy', 'COPD', 'Non-COPD']
EPOCHS = 100
LEARNING_RATE = 1e-4

# ==========================================
# ĐỊNH NGHĨA VITIS AI PREPROCESSING LAYER
# ==========================================
# CRITICAL: Yêu cầu bắt buộc của Vitis AI Model Zoo:
# input = 2 * (original_input / 255.0 - 0.5)
# Chúng ta sẽ sử dụng tf.keras.layers.Rescaling để lồng trực tiếp 
# logic này vào bên trong đồ thị mô hình (đầu Pipeline).

# Khi tf.keras.utils.image_dataset_from_directory load ảnh, nó mặc định trả về giá trị pixel [0, 255].
# Rescaling layer có công thức chung: outputs = inputs * scale + offset
# Dựa trên công thức của Xilinx Vitis-AI: 
#   input = 2 * (pix/255.0) - 2 * 0.5 
#         = pix * (2/255.0) - 1.0
# Suy ra: scale = 2.0 / 255.0, offset = -1.0
vitis_ai_preprocessing = tf.keras.layers.Rescaling(scale=2.0 / 255.0, offset=-1.0, name="vitis_ai_scaler")

# ==========================================
# CHUẨN BỊ (DATA LOADERS)
# ==========================================
print(f"Loading dataset from: {DATASET_DIR}")

# Chia 80% train / 20% validation
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical' # Trả về mảng one-hot cho 3 lớp (3 nodes)
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_dataset.class_names
print(f"Detected Classes: {class_names}")

# Tối ưu hoá luồng (Prefetch) để tránh nghẽn I/O khi đọc ảnh Spectrogram từ Disk
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# ĐỊNH NGHĨA (TRANSFER LEARNING MODEL)
# ==========================================
# Ta xây dựng đầu vào
inputs = Input(shape=IMG_SIZE + (3,)) # (224, 224, 3)

# 1. Custom Preprocessing cho Vitis AI
x = vitis_ai_preprocessing(inputs)

# 2. Xây dựng backbone dựa trên pre-trained weight (Không sử dụng preprocess gốc của Keras)
base_model = MobileNetV2(
    input_tensor=x,         # Nối trực tiếp tensor đã normalize chuẩn Vitis
    alpha=1.0, 
    weights='imagenet', 
    include_top=False       # Cắt bỏ classifier 1000 class gốc
)

# 3. Freeze Backbone (Chỉ học feature extration có sẵn, giúp tránh hư weight khi train mẫu mới)
base_model.trainable = False

# 4. Thêm Global Average Pooling để làm phẳng Feature Map
x = base_model.output
x = GlobalAveragePooling2D(name="gap_features")(x)

# 5. Lắp ráp lớp phân loại cuối cùng với Softmax 
# (Yêu cầu đề bài: exactly 3 nodes for respiratory diseases)
outputs = Dense(NUM_CLASSES, activation='softmax', name="respiratory_classifier")(x)

model = Model(inputs, outputs)

print("===== Model Architecture Summary =====")
model.summary()

# ==========================================
# CẤU HÌNH COMPILER VÀ CALLBACKS
# ==========================================
# Lưu mô hình .h5 nếu Validation Accuracy đạt cao nhất
checkpoint_path = "/home/iec/Parallel_Computing_on_FPGA/python/mobilenetv2_respiratory_best.h5"
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Dừng lại nếu val_accuracy không tăng trong 15 epochs (Tránh Overfitting)
early_stopping_cb = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# ==========================================
# THỰC THI (TRAINING)
# ==========================================
print(f"Bắt đầu Training. Target thư mục mô hình lưu tại: {checkpoint_path}")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

print(f"Huấn luyện hoàn tất! Mô hình đạt chất lượng tốt nhất đã được lưu (SavedModel).")
print(f"Để thực hiện lượng tử hoá (Vai_q_tensorflow) về sau, hãy convert file {checkpoint_path} thành SavedModel (.pb).")
