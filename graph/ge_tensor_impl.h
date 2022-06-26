/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRAPH_GE_TENSOR_IMPL_H_
#define GRAPH_GE_TENSOR_IMPL_H_


#include <deque>
#include <string>
#include <vector>
#include "graph/ge_tensor.h"

namespace ge {
class GeTensorDescImpl {
 public:
  GeTensorDescImpl();
  GeTensorDescImpl(GeShape shape, Format format, DataType dt);
  GeTensorDescImpl(const GeTensorDescImpl &desc);
  GeTensorDescImpl(GeTensorDescImpl &&desc);
  GeTensorDescImpl(const ProtoMsgOwner &proto_owner, proto::TensorDescriptor *proto_msg);
  ~GeTensorDescImpl() = default;

  void Init();
  GeShape &ShapeReference() const;

  bool GeTensorDescAttrsAreEqual(const GeTensorDescImpl &r_ge_tensor_desc) const;
  bool operator==(const GeTensorDescImpl &r_ge_tensor_desc) const;

  GeTensorDescImpl &operator=(const GeTensorDescImpl &desc);
  GeTensorDescImpl &operator=(GeTensorDescImpl &&desc);

  ProtoAttrMapHelper MutableAttrMap();
  ConstProtoAttrMapHelper GetAttrMap() const;
  void SetShape(const GeShape &shape);

  void SetDataType(DataType dataType);
  DataType GetDataType() const;
  void SetFormat(Format format);
  Format GetFormat() const;
  void SetOriginFormat(Format format);
  void SetDeviceType(DeviceType type);
  void SetName(const std::string &name);
  const std::string GetName() const;
  void RefTo(const GeTensorDescImpl &tensorDesc) { tensor_descriptor_ = tensorDesc.tensor_descriptor_; }

 private:
  friend class GeTensorImpl;
  friend class TensorUtils;
  friend class GeAttrValueImp;
  friend class ModelSerializeImp;
  friend class OnnxUtils;
  GeIrProtoHelper<proto::TensorDescriptor> tensor_descriptor_;
  // Reference from tensorDescriptor_, do not direct use
  mutable GeShape __shape_;
};

class TensorDataImpl {
 public:
  TensorDataImpl() = default;

  TensorDataImpl(const TensorDataImpl &other);

  ~TensorDataImpl() = default;

  TensorDataImpl &operator=(const TensorDataImpl &other);

  graphStatus SetData(const uint8_t *data, size_t size);
  graphStatus SetData(uint8_t *data, size_t size, const AlignedPtr::Deleter &delete_fuc);
  void SetData(std::shared_ptr<AlignedPtr> aligned_ptr, size_t size);

  const uint8_t *MallocAlignedPtr(size_t size);

  size_t GetSize() const;
  const uint8_t *GetData() const;
  uint8_t *GetData();

  void clear();

  uint8_t operator[](size_t index) const;

  const std::shared_ptr<AlignedPtr> &GetAlignedPtr() { return aligned_ptr_; }

 private:
  friend class GeTensorImpl;
  friend class TensorUtils;
  friend class GeAttrValueImp;
  friend class ModelSerializeImp;
  GeIrProtoHelper<proto::TensorDescriptor> tensor_descriptor_;
  std::shared_ptr<AlignedPtr> aligned_ptr_ = nullptr;
  size_t length_ = 0;
  // functions data() & mutable_data() return address of invalid_data_ when length_ is 0
  // defined for coding convenience
  static uint32_t invalid_data_;
};

class GeTensorImpl {
 public:
  GeTensorImpl();
  explicit GeTensorImpl(const GeTensorDesc &tensor_desc);
  GeTensorImpl(const GeTensorDesc &tensor_desc, const std::vector<uint8_t> &data);
  GeTensorImpl(const GeTensorDesc &tensor_desc, const uint8_t *data, size_t size);
  GeTensorImpl(GeTensorDesc &&tensor_desc, std::vector<uint8_t> &&data);
  GeTensorImpl(const GeTensorDesc &tensor_desc, const Buffer &data);
  GeTensorImpl(const GeTensorDesc &tensor_desc, std::shared_ptr<AlignedPtr> aligned_ptr, size_t size);
  GeTensorImpl(const GeTensorDesc &tensor_desc, size_t size);
  GeTensorImpl(const ProtoMsgOwner &proto_owner, proto::TensorDef *proto_msg);
  GeTensorImpl(const GeTensorImpl &other);

  ~GeTensorImpl() = default;

  GeTensorImpl &operator=(const GeTensorImpl &other);

  GeTensorDesc &DescReference() const;
  void BuildAlignerPtrWithProtoData();
  graphStatus SetData(std::vector<uint8_t> &&data);
  graphStatus SetData(const std::vector<uint8_t> &data);
  graphStatus SetData(const uint8_t *data, size_t size);
  graphStatus SetData(const Buffer &data);
  graphStatus SetData(const TensorData &data);
  graphStatus SetData(uint8_t *data, size_t size, const AlignedPtr::Deleter &delete_fuc);
  void ClearData();
  void Clone(GeTensorImpl &tensor) const;

  std::shared_ptr<AlignedPtr> GetAlignedPtr();
  const TensorData &GetData() const { return tensor_data_; }
  TensorData &MutableData() { return tensor_data_; }
  // zero copy SetData
  void SetData(std::shared_ptr<AlignedPtr> aligned_ptr, size_t size) {
    tensor_data_.SetData(std::move(aligned_ptr), size);
  }

 private:
  friend class TensorUtils;
  friend class GeAttrValueImp;
  friend class ModelSerializeImp;
  GeIrProtoHelper<proto::TensorDef> tensor_def_;
  // Reference from tensor_data_, do not direct use
  mutable GeTensorDesc __desc_;
  TensorData tensor_data_;
};
}  // namespace ge
#endif  // GRAPH_GE_TENSOR_IMPL_H_
