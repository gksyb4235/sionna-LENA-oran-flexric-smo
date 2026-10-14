// SPDX-License-Identifier: GPL-2.0-only
// Check that changing one cell's RET changes only its cached propagation.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/antenna-module.h"
#include "ns3/sionna-lookup-channel-model.h"
#include <pybind11/embed.h>
#include <iostream>
#include <iomanip>
using namespace ns3;
namespace py=pybind11;

static Ptr<MobilityModel> At(Vector position)
{
    auto node=CreateObject<Node>();
    auto mobility=CreateObject<ConstantPositionMobilityModel>();
    node->AggregateObject(mobility);mobility->SetPosition(position);return mobility;
}
static Ptr<UniformPlanarArray> Array(int side)
{
    auto array=CreateObject<UniformPlanarArray>();
    array->SetAttribute("NumRows",UintegerValue(side));array->SetAttribute("NumColumns",UintegerValue(side));
    return array;
}
static double Power(Ptr<const MatrixBasedChannelModel::ChannelMatrix> channel)
{
    double power=0;
    const auto& values=channel->m_channel;
    for(size_t i=0;i<values.GetNumRows();++i)
        for(size_t j=0;j<values.GetNumCols();++j)
            for(size_t k=0;k<values.GetNumPages();++k)power+=std::norm(values(i,j,k));
    return power;
}
int main(int argc,char** argv)
{
    try {
    py::scoped_interpreter interpreter{};
    std::string root="scenarios/khu-real/ret_caches";
    CommandLine cmd;cmd.AddValue("root","RET cache root",root);cmd.Parse(argc,argv);
    auto low=CreateObject<SionnaLookupChannelModel>();low->SetFrequency(1.8e9);
    low->SetCacheFile(root+"/tilt_3deg/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5");
    auto high=CreateObject<SionnaLookupChannelModel>();high->SetFrequency(3.5e9);
    high->SetCacheFile(root+"/tilt_3deg/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5");
    py::object f=py::module_::import("h5py").attr("File")(low->GetCacheFile(),"r");
    py::object positions=f["gNB_4G_1"]["positions"].attr("__getitem__")(6335);
    Vector position(py::cast<double>(positions[py::int_(0)]),py::cast<double>(positions[py::int_(1)]),py::cast<double>(positions[py::int_(2)]));
    f.attr("close")();
    auto ue=At(position),one=At(Vector(-326.9,406.6,57.23)),two=At(Vector(-381.23,219.78,67.5)),three=At(Vector(-360.04,82.73,82.62));
    auto ua=Array(1),a=Array(2),b=Array(2),c=Array(2);
    auto measure=[&](){return std::vector<double>{Power(low->GetChannel(one,ue,a,ua)),Power(low->GetChannel(two,ue,b,ua)),Power(high->GetChannel(three,ue,c,ua))};};
    const auto before=measure();
    std::cout<<"RET_CACHE_BASELINE "<<before[0]<<","<<before[1]<<","<<before[2]<<std::endl;
    NS_ABORT_MSG_IF(!low->PrepareRetCache("gNB_4G_1",9,root),"Cannot prepare cell1 RET9");
    NS_ABORT_MSG_IF(!low->PrepareRetCache("gNB_4G_1",3,root),"Cannot prepare restoration");
    NS_ABORT_MSG_IF(low->PrepareRetCache("gNB_4G_1",6,root),"Missing cache was accepted");
    Simulator::Schedule(Seconds(1),[&](){
        low->ActivateRetCache("gNB_4G_1",9);
        const auto changed=measure();
        NS_ABORT_MSG_IF(std::abs(changed[0]-before[0])<=std::abs(before[0])*1e-8,"Target channel did not change");
        NS_ABORT_MSG_IF(changed[1]!=before[1] || changed[2]!=before[2],"Unchanged cell channel changed");
        low->ActivateRetCache("gNB_4G_1",3);
        const auto restored=measure();
        NS_ABORT_MSG_IF(restored!=before,"Restoration did not recover original channel");
        std::cout<<std::setprecision(16)<<"RET_CACHE_PASS before="<<before[0]<<" after="<<changed[0]<<" restored="<<restored[0]<<" unchanged_cells=2"<<std::endl;
    });
    Simulator::Stop(Seconds(2));Simulator::Run();Simulator::Destroy();
    return 0;
    } catch(const std::exception& error) {std::cerr<<"RET_CACHE_ERROR "<<error.what()<<std::endl;return 1;}
}
