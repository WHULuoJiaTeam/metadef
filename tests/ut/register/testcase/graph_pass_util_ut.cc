#include <gtest/gtest.h>
#define private public
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"

using namespace std;
using namespace ge;

namespace fe {
class GraphPassUtilUT : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(GraphPassUtilUT, set_output_desc_attr_case1) {
  NodePtr origin_node = nullptr;
  NodePtr fusion_node = nullptr;
  GraphPassUtil::SetOutputDescAttr(0, 0, origin_node, fusion_node);
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case2) {
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);
  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  GraphPassUtil::SetOutputDescAttr(1, 0, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), false);
  GraphPassUtil::SetOutputDescAttr(0, 1, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), false);
  GraphPassUtil::SetOutputDescAttr(0, 0, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), true);
  string origin_name;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_name);
  EXPECT_EQ(origin_name, "relu1");
  string origin_dtype;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_dtype);
  EXPECT_EQ(origin_dtype, "DT_FLOAT");
  string origin_format;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format);
  EXPECT_EQ(origin_format, "NCHW");
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case3) {
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, "origin_relu1");
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, "DT_DOUBLE");
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, "ND");
  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);

  GraphPassUtil::SetOutputDescAttr(0, 0, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), true);
  string origin_name;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_name);
  EXPECT_EQ(origin_name, "origin_relu1");
  string origin_dtype;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_dtype);
  EXPECT_EQ(origin_dtype, "DT_DOUBLE");
  string origin_format;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format);
  EXPECT_EQ(origin_format, "ND");
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case4) {
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);
  vector<string> names = {"ori_rule1"};
  AttrUtils::SetListStr(relu1, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names);
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, "RESERVED");
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, "RESERVED");
  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);

  GraphPassUtil::SetOutputDescAttr(0, 0, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), true);
  string origin_name;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_name);
  EXPECT_EQ(origin_name, "ori_rule1");
  string origin_dtype;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_dtype);
  EXPECT_EQ(origin_dtype, "DT_FLOAT");
  string origin_format;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format);
  EXPECT_EQ(origin_format, "NCHW");
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case5) {

}

}
