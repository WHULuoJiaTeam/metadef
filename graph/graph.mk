LOCAL_PATH := $(call my-dir)
include $(LOCAL_PATH)/stub/Makefile
COMMON_LOCAL_SRC_FILES := \
    ./proto/om.proto \
    ./proto/ge_ir.proto \
    ./proto/ge_onnx.proto \
    ./proto/insert_op.proto \
    ./proto/task.proto \
    ./proto/fwk_adapter.proto \
    ./proto/op_mapping_info.proto \
    ./proto/dump_task.proto \
    ./anchor.cc \
    ./ge_attr_value.cc \
    ./attr_value.cc \
    ./buffer.cc \
    ./aligned_ptr.cc \
    ./compute_graph.cc \
    ./ascend_string.cc \
    ./gnode.cc \
    ./graph.cc \
    ./inference_context.cc \
    ./shape_refiner.cc \
    ./format_refiner.cc \
    ./ref_relation.cc \
    ./model.cc \
    ./model_serialize.cc \
    ./node.cc \
    ./op_desc.cc \
    ./operator.cc \
    ./operator_factory.cc \
    ./operator_factory_impl.cc \
    ./ge_attr_define.cc \
    ./ge_tensor.cc \
    ./detail/attributes_holder.cc \
    ./utils/anchor_utils.cc \
    ./utils/tuning_utils.cc \
    ./utils/graph_utils.cc \
    ./utils/dumper/ge_graph_dumper.cc \
    ./utils/ge_ir_utils.cc \
    ./utils/op_desc_utils.cc \
    ./utils/type_utils.cc \
    ./utils/tensor_utils.cc \
    ./tensor.cc \
    ./debug/graph_debug.cc \
    ./opsproto/opsproto_manager.cc \
    ../ops/op_imp.cpp \
    option/ge_context.cc \
    option/ge_local_context.cc \
    ./runtime_inference_context.cc \
    ./utils/node_utils.cc \
    ../third_party/transformer/src/axis_util.cc \
    ../third_party/transformer/src/transfer_shape_according_to_format.cc \
    ../third_party/transformer/src/expand_dimension.cc \
    ./utils/transformer_utils.cc \


COMMON_LOCAL_C_INCLUDES := \
    proto/om.proto \
    proto/ge_ir.proto \
    proto_inner/ge_onnx.proto \
    proto/insert_op.proto \
    proto/task.proto \
    proto/fwk_adapter.proto \
    proto/op_mapping_info.proto \
    proto/dump_task.proto \
    inc \
    metadef/inc \
    graphengine/inc \
    inc/external \
    metadef/inc/external \
    graphengine/inc/external \
    metadef/inc/external/graph \
    metadef/inc/graph \
    metadef/inc/common \
    metadef \
    metadef/graph \
    third_party/protobuf/include \
    $(TOPDIR)metadef/third_party \
    $(TOPDIR)metadef/third_party/transformer/inc \
    libc_sec/include \
    ops/built-in/op_proto/inc \
    cann/ops/built-in/op_proto/inc \


#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS += -DFMK_SUPPORT_DUMP -O2 -Dgoogle=ascend_private -Wno-deprecated-declarations
LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libascend_protobuf   \
    libslog       \
    liberror_manager \

LOCAL_STATIC_LIBRARIES := \
    libmmpa       \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_SHARED_LIBRARY)

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := stub/libgraph

LOCAL_CFLAGS += -DFMK_SUPPORT_DUMP -O2 -Wno-deprecated-declarations
LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := \
    ../../out/graph/lib64/stub/attr_value.cc \
    ../../out/graph/lib64/stub/graph.cc \
    ../../out/graph/lib64/stub/operator.cc \
    ../../out/graph/lib64/stub/tensor.cc \
    ../../out/graph/lib64/stub/operator_factory.cc \
    ../../out/graph/lib64/stub/ascend_string.cc \
    ../../out/graph/lib64/stub/gnode.cc \

LOCAL_SHARED_LIBRARIES :=

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_SHARED_LIBRARY)

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := fwk_stub/libgraph

LOCAL_CFLAGS += -DFMK_SUPPORT_DUMP -O2 -Wno-deprecated-declarations
LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := \
    ../../out/graph/lib64/stub/attr_value.cc \
    ../../out/graph/lib64/stub/graph.cc \
    ../../out/graph/lib64/stub/operator.cc \
    ../../out/graph/lib64/stub/operator_factory.cc \
    ../../out/graph/lib64/stub/tensor.cc \
    ../../out/graph/lib64/stub/inference_context.cc \
    ../../out/graph/lib64/stub/ascend_string.cc \
    ../../out/graph/lib64/stub/gnode.cc \

LOCAL_SHARED_LIBRARIES :=

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_SHARED_LIBRARY)

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private -Wno-deprecated-declarations

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libascend_protobuf   \
    libslog       \
    liberror_manager \

LOCAL_STATIC_LIBRARIES := \
    libmmpa       \

LOCAL_LDFLAGS := -lrt -ldl

ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
endif

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_SHARED_LIBRARY)

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := stub/libgraph

LOCAL_CFLAGS += -O2

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := \
    ../../out/graph/lib64/stub/attr_value.cc \
    ../../out/graph/lib64/stub/graph.cc \
    ../../out/graph/lib64/stub/operator.cc \
    ../../out/graph/lib64/stub/tensor.cc \
    ../../out/graph/lib64/stub/operator_factory.cc \
    ../../out/graph/lib64/stub/ascend_string.cc \
    ../../out/graph/lib64/stub/gnode.cc \

LOCAL_SHARED_LIBRARIES :=

LOCAL_LDFLAGS := -lrt -ldl

ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
endif

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_SHARED_LIBRARY)

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := fwk_stub/libgraph

LOCAL_CFLAGS += -O2

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := \
    ../../out/graph/lib64/stub/attr_value.cc \
    ../../out/graph/lib64/stub/graph.cc \
    ../../out/graph/lib64/stub/operator.cc \
    ../../out/graph/lib64/stub/operator_factory.cc \
    ../../out/graph/lib64/stub/tensor.cc \
    ../../out/graph/lib64/stub/inference_context.cc \
    ../../out/graph/lib64/stub/ascend_string.cc \
    ../../out/graph/lib64/stub/gnode.cc \


LOCAL_SHARED_LIBRARIES :=

LOCAL_LDFLAGS := -lrt -ldl

ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
endif

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_SHARED_LIBRARY)

# compile for ut/st
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS += -Dgoogle=ascend_private

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libascend_protobuf   \
    libslog       \
    liberror_manager \

LOCAL_STATIC_LIBRARIES := \
    libmmpa       \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_LLT_SHARED_LIBRARY)


#compiler for host static lib
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS += -DFMK_SUPPORT_DUMP -O2 -Dgoogle=ascend_private
LOCAL_CPPFLAGS += -fexceptions

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_STATIC_LIBRARIES := \
    libascend_protobuf   \

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libslog       \
    liberror_manager \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_HOST_STATIC_LIBRARY)

#compiler for device static lib
include $(CLEAR_VARS)
LOCAL_MODULE := libgraph

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)
LOCAL_SRC_FILES  := $(COMMON_LOCAL_SRC_FILES)

LOCAL_STATIC_LIBRARIES := \
    libascend_protobuf   \

LOCAL_SHARED_LIBRARIES := \
    libc_sec      \
    libslog       \
    liberror_manager \

LOCAL_LDFLAGS := -lrt -ldl

LOCAL_MULTILIB := 64
LOCAL_PROPRIETARY_MODULE := true

include $(BUILD_STATIC_LIBRARY)
