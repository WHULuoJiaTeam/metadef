LOCAL_PATH := $(call my-dir)


local_lib_src_files :=  error_manager.cc \


local_lib_inc_path := \
    inc \
    metadef/inc \
    graphengine/inc \
    inc/external \
    metadef/inc/external \
    graphengine/inc/external \
    libc_sec/include \
    third_party/json/include \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := liberror_manager

LOCAL_CFLAGS += -Werror

LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS := -lrt -ldl

LOCAL_WHOLE_STATIC_LIBRARIES := libmmpa \

LOCAL_SHARED_LIBRARIES := libslog \
                          libc_sec \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_SHARED_LIBRARY}

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := liberror_manager

LOCAL_CFLAGS += -Werror

LOCAL_CFLAGS += -std=c++11
ifeq ($(device_os),android)
LOCAL_LDFLAGS := -ldl
else
LOCAL_LDFLAGS := -lrt -ldl
endif

LOCAL_WHOLE_STATIC_LIBRARIES := libmmpa \

LOCAL_SHARED_LIBRARIES := libslog \
                          libc_sec \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include $(BUILD_SHARED_LIBRARY)

#compiler static liberror_manager for host
include $(CLEAR_VARS)
LOCAL_MODULE := liberror_manager

LOCAL_CFLAGS += -Werror

LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES := libslog \
                          libc_sec \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

LOCAL_UNINSTALLABLE_MODULE := false

include ${BUILD_HOST_STATIC_LIBRARY}

#compiler static liberror_manager for device
include $(CLEAR_VARS)
LOCAL_MODULE := liberror_manager

LOCAL_CFLAGS += -Werror

LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES := libslog \
                          libc_sec \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

LOCAL_UNINSTALLABLE_MODULE := false
include ${BUILD_STATIC_LIBRARY}

#compiler for llt
include $(CLEAR_VARS)
LOCAL_MODULE := liberror_manager

LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS := -lrt -ldl

LOCAL_WHOLE_STATIC_LIBRARIES :=   libmmpa \

LOCAL_SHARED_LIBRARIES :=   libslog \
                            libc_sec \

LOCAL_SRC_FILES := $(local_lib_src_files)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_LLT_SHARED_LIBRARY}
