PYTHON ?= python3
CMAKE ?= cmake
BUILD_TYPE ?= Release
ARTIFACT_ROOT ?= artifacts
TRAINING_ARTIFACTS ?= $(ARTIFACT_ROOT)/training
QUANTIZATION_ARTIFACTS ?= $(ARTIFACT_ROOT)/quantization
COMPILE_ARTIFACTS ?= $(ARTIFACT_ROOT)/compile
PAPER_FIGURES ?= $(ARTIFACT_ROOT)/paper_figures
WANDB ?= --wandb
CMAKE_BUILD_DIR ?= $(COMPILE_ARTIFACTS)/cmake-$(BUILD_TYPE)

PY_FILES := $(shell find python fpga -name '*.py' -type f | sort)

.PHONY: help dirs build build-debug build-release test test-cpp run py-compile \
	preprocess-combined preprocess-audio wavelet train train-mobilenet \
	train-efficientnet distill train-kd-bilstm train-kd-icbhi train-kd-icbhi-effnet train-kd-teachers train-kd-students \
	calib-data quantize quantize-vitis quantize-flow plot-signal plot-kd \
	plot-teacher-student plot-all clean-compile

help:
	@printf "Respiratory Sound Analysis runner\n\n"
	@printf "C++ targets:\n"
	@printf "  make build              Configure and build C++ into artifacts/compile\n"
	@printf "  make build-debug        Build C++ Debug configuration\n"
	@printf "  make build-release      Build C++ Release configuration\n"
	@printf "  make test-cpp           Run C++ tests with ctest\n"
	@printf "  make run                Run the C++ demo executable\n\n"
	@printf "Python pipeline:\n"
	@printf "  make py-compile         Syntax-check all Python files\n"
	@printf "  make preprocess-combined Combine source datasets into data/combined\n"
	@printf "  make preprocess-audio   Preprocess combined audio\n"
	@printf "  make wavelet            Generate CWT spectrograms\n"
	@printf "  make train-mobilenet    Train MobileNetV2 into artifacts/training\n"
	@printf "  make train-efficientnet Train EfficientNet-B0 into artifacts/training\n"
	@printf "  make distill            Run legacy active distillation pipeline\n"
	@printf "  make train-kd-bilstm    Run CNN-BiLSTM → CNN6 KD pipeline (ICBHI 2017)\n"
	@printf "  make train-kd-icbhi     Train ICBHI 3-class teacher ensemble + CNN student\n"
	@printf "  make train-kd-icbhi-effnet Train ICBHI 3-class disease EfficientNet-B0\n"
	@printf "  make train-kd-teachers  Train only ICBHI 3-class teacher ensemble\n"
	@printf "  make train-kd-students  Train only ICBHI 3-class KD students\n"
	@printf "                         KD targets use WandB by default; disable with WANDB=\n"
	@printf "  make calib-data         Generate calibration data\n"
	@printf "  make quantize-vitis     Run active Vitis QAT quantization\n"
	@printf "  make quantize-flow      Run fpga/vitis_ai_flow quantizer\n\n"
	@printf "Figures:\n"
	@printf "  make plot-signal        Generate signal figure\n"
	@printf "  make plot-kd            Generate KD architecture figure\n"
	@printf "  make plot-teacher-student Generate teacher/student figure\n"
	@printf "  make plot-all           Run all figure targets\n"

dirs:
	mkdir -p "$(TRAINING_ARTIFACTS)" "$(QUANTIZATION_ARTIFACTS)" "$(COMPILE_ARTIFACTS)" "$(PAPER_FIGURES)"

build: dirs
	$(CMAKE) -S . -B "$(CMAKE_BUILD_DIR)" -DCMAKE_BUILD_TYPE="$(BUILD_TYPE)"
	$(CMAKE) --build "$(CMAKE_BUILD_DIR)"

build-debug:
	$(MAKE) build BUILD_TYPE=Debug CMAKE_BUILD_DIR=$(COMPILE_ARTIFACTS)/cmake-debug

build-release:
	$(MAKE) build BUILD_TYPE=Release CMAKE_BUILD_DIR=$(COMPILE_ARTIFACTS)/cmake-release

test: test-cpp

test-cpp: build
	ctest --test-dir "$(CMAKE_BUILD_DIR)" --output-on-failure

run: build
	"$(CMAKE_BUILD_DIR)/respiratory_analysis"

py-compile:
	$(PYTHON) -m py_compile $(PY_FILES)

preprocess-combined:
	$(PYTHON) python/preprocessing/combine_dataset.py --output-dir data/combined

preprocess-audio:
	$(PYTHON) python/preprocessing/preprocessing.py --data_dir data/combined/audio --labels_csv data/combined/labels.csv --output_dir data/combined/processed_audio

wavelet:
	$(PYTHON) python/preprocessing/wavelet_transform.py --data_dir data/combined/processed_audio --output_dir data/combined/spectrograms

train: train-mobilenet

train-mobilenet:
	$(PYTHON) python/training/train_mobilenetv2.py --artifact_root "$(TRAINING_ARTIFACTS)"

train-efficientnet:
	$(PYTHON) python/training/train_efficientnet_b0.py --artifact_root "$(TRAINING_ARTIFACTS)"

distill:
	$(PYTHON) python/training/distillation_02.py --artifact_root "$(TRAINING_ARTIFACTS)"

train-kd-bilstm:
	$(PYTHON) python/training/kd_cnn_bilstm_to_cnn.py --artifact_root "$(TRAINING_ARTIFACTS)" $(WANDB)

train-kd-icbhi:
	$(PYTHON) python/training/kd_icbhi_3class.py --stage all --artifact_root "$(ARTIFACT_ROOT)" $(WANDB)

train-kd-icbhi-effnet:
	$(PYTHON) python/training/kd_icbhi_4class_efficientnet.py --artifact_root "$(ARTIFACT_ROOT)" $(WANDB)

train-kd-teachers:
	$(PYTHON) python/training/kd_icbhi_3class.py --stage teachers --artifact_root "$(ARTIFACT_ROOT)" $(WANDB)

train-kd-students:
	$(PYTHON) python/training/kd_icbhi_3class.py --stage students --artifact_root "$(ARTIFACT_ROOT)" $(WANDB)

calib-data:
	$(PYTHON) python/quantization/generate_calib_data.py --artifact_root "$(ARTIFACT_ROOT)"

quantize: quantize-vitis

quantize-vitis:
	$(PYTHON) python/quantization/quantize_distillation_03.py --artifact_root "$(ARTIFACT_ROOT)"

quantize-flow:
	$(PYTHON) fpga/vitis_ai_flow/quantize.py --artifact_root "$(ARTIFACT_ROOT)"

plot-signal:
	$(PYTHON) python/visualizations/visualize_signal.py --output_dir "$(PAPER_FIGURES)"

plot-kd:
	$(PYTHON) python/visualizations/plot_kd_architecture.py --output_dir "$(PAPER_FIGURES)"

plot-teacher-student:
	$(PYTHON) python/visualizations/plot_teacher_student_nn.py --output_dir "$(PAPER_FIGURES)"

plot-all: plot-signal plot-kd plot-teacher-student

clean-compile:
	rm -rf "$(COMPILE_ARTIFACTS)"/cmake-*
