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
#ifndef INC_89E3042EF25240149AD95BEE22C88126
#define INC_89E3042EF25240149AD95BEE22C88126

#include "graph/compute_graph.h"

namespace ge {
struct GeGraphDumper {
  virtual void Dump(const ge::ComputeGraphPtr &graph, const std::string &suffix){}
  virtual ~GeGraphDumper() {}
};

struct GraphDumperRegistry {
  static GeGraphDumper &GetDumper();
  static void Register(GeGraphDumper &);
};

}  // namespace ge

#endif
