// SPDX-License-Identifier: GPL-2.0-only
// Real FlexRIC/E2 integration fixture; standard NR channel, no Sionna inference.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/nr-module.h"
#include "ns3/antenna-module.h"
#include "ns3/agora-ric-controller.h"
#include <chrono>
#include <thread>
using namespace ns3;

static void Pace(uint32_t milliseconds)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    Simulator::Schedule(Seconds(1),&Pace,milliseconds);
}
int main(int argc,char** argv)
{
    double duration=60;
    uint32_t paceMs=200;
    double txp5g=43,txp4g1=43,txp4g2=43,ret5g=3,ret4g1=3,ret4g2=3;
    double cio5g=5,cio4g1=0,cio4g2=0,ttt=256,hys=3;
    CommandLine cmd;
    cmd.AddValue("simTime","Simulation seconds",duration);
    cmd.AddValue("paceMs","Wall-clock milliseconds per simulation second",paceMs);
    cmd.AddValue("txp_5g","5g TxP",txp5g);cmd.AddValue("txp_4g1","4g1 TxP",txp4g1);cmd.AddValue("txp_4g2","4g2 TxP",txp4g2);
    cmd.AddValue("ret_5g","5g RET",ret5g);cmd.AddValue("ret_4g1","4g1 RET",ret4g1);cmd.AddValue("ret_4g2","4g2 RET",ret4g2);
    cmd.AddValue("cio_5g","5g CIO",cio5g);cmd.AddValue("cio_4g1","4g1 CIO",cio4g1);cmd.AddValue("cio_4g2","4g2 CIO",cio4g2);
    cmd.AddValue("ttt_ms","Global TTT",ttt);cmd.AddValue("hys_db","Global HYS",hys);
    cmd.Parse(argc,argv);
    Config::SetDefault("ns3::NrHelper::E2ModeNr",BooleanValue(true));
    Config::SetDefault("ns3::NrHelper::E2TermIp",StringValue("127.0.0.1"));
    Config::SetDefault("ns3::NrHelper::E2LocalPort",UintegerValue(39200));
    auto nr=CreateObject<NrHelper>();
    nr->SetAttribute("UseIdealRrc",BooleanValue(true));
    auto epc=CreateObject<NrPointToPointEpcHelper>();
    nr->SetEpcHelper(epc);
    auto beam=CreateObject<IdealBeamformingHelper>();
    beam->SetAttribute("BeamformingMethod",TypeIdValue(DirectPathBeamforming::GetTypeId()));
    nr->SetBeamformingHelper(beam);
    nr->SetHandoverAlgorithmType("ns3::NrA3RsrpHandoverAlgorithm");
    nr->SetHandoverAlgorithmAttribute("TimeToTrigger",TimeValue(MilliSeconds(ttt)));
    nr->SetHandoverAlgorithmAttribute("Hysteresis",DoubleValue(hys));
    nr->SetGnbAntennaAttribute("NumRows",UintegerValue(2));
    nr->SetGnbAntennaAttribute("NumColumns",UintegerValue(2));
    nr->SetUeAntennaAttribute("NumRows",UintegerValue(1));
    nr->SetUeAntennaAttribute("NumColumns",UintegerValue(1));
    NodeContainer cells,ues;cells.Create(3);ues.Create(1);
    MobilityHelper mobility;mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(cells);mobility.Install(ues);
    for(uint32_t i=0;i<3;++i)cells.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(i*100,0,20));
    ues.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(10,0,1.5));
    CcBwpCreator creator;
    CcBwpCreator::SimpleOperationBandConf bandConf(3.5e9,10e6,1);
    auto band=creator.CreateOperationBandContiguousCc(bandConf);
    auto channel=CreateObject<NrChannelHelper>();
    channel->ConfigureFactories("UMa","Default","ThreeGpp");
    channel->AssignChannelsToBands({band});
    auto bwps=CcBwpCreator::GetAllBwps({band});
    auto gnbs=nr->InstallGnbDevice(cells,bwps);
    auto devices=nr->InstallUeDevice(ues,bwps);
    const std::vector<std::string> labels{"5g","4g1","4g2"};
    const std::vector<double> powers{txp5g,txp4g1,txp4g2},tilts{ret5g,ret4g1,ret4g2},cios{cio5g,cio4g1,cio4g2};
    std::map<std::string,Ptr<NrGnbNetDevice>> mapping;
    for(uint32_t i=0;i<3;++i)
    {
        auto dev=DynamicCast<NrGnbNetDevice>(gnbs.Get(i));mapping[labels[i]]=dev;
        dev->GetPhy(0)->SetTxPower(powers[i]);
        auto antenna=DynamicCast<UniformPlanarArray>(dev->GetPhy(0)->GetSpectrumPhy()->GetAntenna());
        antenna->SetAttribute("DowntiltAngle",DoubleValue(tilts[i]*M_PI/180.0));
        DynamicCast<NrUeNetDevice>(devices.Get(0))->GetRrc()->SetCellIndividualOffset(dev->GetCellId(),cios[i]);
    }
    InternetStackHelper internet;internet.Install(ues);
    epc->AssignUeIpv4Address(devices);
    nr->AddX2Interface(cells);
    nr->AttachToClosestGnb(devices,gnbs);
    AgoraRicController::InstallFromEnvironment(mapping,duration);
    Simulator::Schedule(Seconds(0),&Pace,paceMs);
    Simulator::Stop(Seconds(duration));Simulator::Run();Simulator::Destroy();
}
