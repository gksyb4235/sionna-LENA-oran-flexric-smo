// Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

/**
 * @file khu-real-nr-sionna.cc
 * @brief NR + official Sionna RT channel model against the real KHU campus
 * scene, with real gNB positions and a real moving UE trace read from CSV.
 *
 * This is the NR-module equivalent of ns-O-RAN-flexric's
 * scratch/scenario-zero-sionna-kyunghee.cc (legacy mmwave module + TU
 * Berlin ZMQ/protobuf Sionna bridge). The CSV parsing (gNB positions,
 * UE trace samples) is ported near-verbatim from that file; the
 * NR/channel/traffic setup is adapted from contrib/nr/examples/
 * cttc-nr-demo-sionna-rt.cc, replacing its GridScenarioHelper static
 * topology with the CSV-driven one below.
 *
 * gNB CSV: gnb_id,x,y,z[,bearing_deg,tilt_deg]
 * UE trace CSV: time_s,ue_id,x,y,z,active
 *
 * RET (bearing_deg/tilt_deg) is parsed but not yet applied to the NR
 * antenna model -- that dispatch path doesn't exist on the NR side yet
 * (see NR_ROADMAP.md-equivalent notes); this scenario is for topology
 * and raw simulation speed, not RET validation.
 */

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/sionna-rt-channel-model.h"
#include "ns3/sionna-rt-spectrum-propagation-loss-model.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("KhuRealNrSionna");

struct UeTraceSample
{
    double time{0.0};
    Vector position;
    bool active{true};
};

struct UeTrace
{
    std::string externalId;
    std::vector<UeTraceSample> samples;
};

struct GnbPosition
{
    std::string externalId;
    Vector position;
    double bearingDeg{std::numeric_limits<double>::quiet_NaN()};
    double tiltDeg{std::numeric_limits<double>::quiet_NaN()};
};

static std::string
Trim(std::string value)
{
    const auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

static bool
ParseActive(const std::string& text)
{
    std::string value = Trim(text);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return std::tolower(ch);
    });
    if (value == "1" || value == "true" || value == "yes" || value == "active")
    {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "inactive")
    {
        return false;
    }
    throw std::runtime_error("invalid active value: " + text);
}

static std::vector<UeTrace>
LoadUeTrace(const std::string& path, uint32_t maxUes)
{
    std::ifstream input(path);
    NS_ABORT_MSG_IF(!input, "Cannot open UE trace: " << path);

    std::vector<UeTrace> traces;
    std::map<std::string, uint32_t> indexById;
    std::string line;
    uint32_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        line = Trim(line);
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::vector<std::string> fields;
        std::stringstream row(line);
        std::string field;
        while (std::getline(row, field, ','))
        {
            fields.push_back(Trim(field));
        }
        if (!fields.empty() && (fields[0] == "time" || fields[0] == "time_s"))
        {
            continue;
        }
        NS_ABORT_MSG_IF(fields.size() != 6,
                        "UE trace line " << lineNumber << " must be time_s,ue_id,x,y,z,active");

        try
        {
            const std::string& externalId = fields[1];
            auto [it, inserted] = indexById.emplace(externalId, traces.size());
            if (inserted)
            {
                NS_ABORT_MSG_IF(traces.size() >= maxUes,
                                "UE trace has more UE IDs than requested UE count " << maxUes);
                traces.push_back(UeTrace{externalId, {}});
            }
            UeTraceSample sample;
            sample.time = std::stod(fields[0]);
            sample.position =
                Vector(std::stod(fields[2]), std::stod(fields[3]), std::stod(fields[4]));
            sample.active = ParseActive(fields[5]);
            traces[it->second].samples.push_back(sample);
        }
        catch (const std::exception& error)
        {
            NS_ABORT_MSG("Invalid UE trace line " << lineNumber << ": " << error.what());
        }
    }

    NS_ABORT_MSG_IF(traces.empty(), "UE trace contains no samples: " << path);
    for (auto& trace : traces)
    {
        std::stable_sort(trace.samples.begin(), trace.samples.end(),
                         [](const auto& a, const auto& b) { return a.time < b.time; });
    }
    return traces;
}

static std::vector<GnbPosition>
LoadGnbPositions(const std::string& path)
{
    std::ifstream input(path);
    NS_ABORT_MSG_IF(!input, "Cannot open gNB position CSV: " << path);

    std::vector<GnbPosition> positions;
    std::string line;
    uint32_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        line = Trim(line);
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::vector<std::string> fields;
        std::stringstream row(line);
        std::string field;
        while (std::getline(row, field, ','))
        {
            fields.push_back(Trim(field));
        }
        if (!fields.empty() && (fields[0] == "gnb_id" || fields[0] == "id"))
        {
            continue;
        }
        NS_ABORT_MSG_IF(fields.size() != 4 && fields.size() != 6,
                        "gNB position line " << lineNumber
                                             << " must be gnb_id,x,y,z[,bearing_deg,tilt_deg]");
        try
        {
            GnbPosition gnb{fields[0], Vector(std::stod(fields[1]), std::stod(fields[2]),
                                              std::stod(fields[3]))};
            if (fields.size() == 6)
            {
                gnb.bearingDeg = std::stod(fields[4]);
                gnb.tiltDeg = std::stod(fields[5]);
            }
            positions.push_back(gnb);
        }
        catch (const std::exception& error)
        {
            NS_ABORT_MSG("Invalid gNB position line " << lineNumber << ": " << error.what());
        }
    }
    NS_ABORT_MSG_IF(positions.empty(), "gNB position CSV contains no positions: " << path);
    return positions;
}


static Vector
InitialPosition(const UeTrace& trace)
{
    Vector position = trace.samples.front().position;
    for (const auto& sample : trace.samples)
    {
        if (sample.time > 0.0)
        {
            break;
        }
        position = sample.position;
    }
    return position;
}

static void
SetSionnaRtPathSolverConfig(const BandwidthPartInfoPtrVector& allBwps,
                            const SionnaRtChannelModel::RtPathSolverConfig& rtPathSolverConfig)
{
    for (const auto& bwp : allBwps)
    {
        Ptr<SpectrumChannel> spectrumChannel = bwp.get()->GetChannel();
        if (!spectrumChannel)
        {
            continue;
        }
        Ptr<PhasedArraySpectrumPropagationLossModel> phasedArrayChannel =
            spectrumChannel->GetPhasedArraySpectrumPropagationLossModel();
        if (!phasedArrayChannel)
        {
            continue;
        }
        Ptr<SionnaRtSpectrumPropagationLossModel> sionna =
            phasedArrayChannel->GetObject<SionnaRtSpectrumPropagationLossModel>();
        if (sionna)
        {
            sionna->SetRtPathSolverConfig(rtPathSolverConfig);
        }
    }
}

// ---- Optional live GUI bridge ----
//
// Reuses ns-O-RAN-flexric's Polyscope GUI ZMQ protocol as-is: scratch/
// zmq_bridge.py here is an unmodified copy of contrib/sionna/gui/src/
// sionna_rt_gui/zmq_bridge.py's ZMQBridgeClient (it has zero dependencies
// on the rest of that package -- json/threading/time/zmq only -- so it's
// copied standalone rather than imported through sionna_rt_gui, which
// would otherwise pull in that package's __init__.py and, transitively,
// polyscope). The GUI process (run.py, its own .venv, unmodified) must
// already be running and listening on guiHost:5600/5601 -- see
// ns-O-RAN-flexric/README.md Terminal 1. This scenario just takes over the
// "publisher" role that kyunghee_server.py used to play: same protocol,
// same ports, different process pushing the updates.

static py::object
ConnectGuiZmqBridge(const std::string& guiSrc, const std::string& guiHost)
{
    if (guiSrc.empty())
    {
        return py::none();
    }
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, guiSrc);
    py::object client = py::module_::import("zmq_bridge").attr("ZMQBridgeClient")(
        py::arg("host") = guiHost);
    client.attr("connect")();
    NS_LOG_UNCOND("[gui] connected to Polyscope GUI ZMQ bridge at " << guiHost);
    return client;
}

static void
SendGnbPositionToGui(py::object& client, const std::string& name, const Vector& pos)
{
    if (client.is_none())
    {
        return;
    }
    try
    {
        client.attr("send_gnb_position")(name, py::make_tuple(pos.x, pos.y, pos.z));
    }
    catch (const py::error_already_set& error)
    {
        NS_LOG_UNCOND("[gui] send_gnb_position failed (ignoring): " << error.what());
    }
}

static void
SendUePositionToGui(py::object& client, const std::string& name, const Vector& pos)
{
    if (client.is_none())
    {
        return;
    }
    try
    {
        client.attr("send_ue_position")(name, py::make_tuple(pos.x, pos.y, pos.z));
    }
    catch (const py::error_already_set& error)
    {
        NS_LOG_UNCOND("[gui] send_ue_position failed (ignoring): " << error.what());
    }
}

int
main(int argc, char* argv[])
{
    py::scoped_interpreter guard{};

    std::string sionnaScene =
        "/home/user/LENA-oran-flexric-smo/scenes/khu-real/KHU_Cropped_Sionna_RT.xml";
    std::string gnbPositionsPath = "/home/user/LENA-oran-flexric-smo/scenarios/khu-real/gnbs-ret.csv";
    std::string sumoTracePath = "/home/user/LENA-oran-flexric-smo/scenarios/khu-real/ues-15.csv";
    uint32_t nUes = 15;

    Time simTime = Seconds(180);
    Time trafficStart = Seconds(0.5);
    bool requireTraffic = false;
    Time sionnaUpdatePeriod = MilliSeconds(50);

    double centralFrequencyBand1 = 3.5e9;
    double bandwidthBand1 = 20e6;
    uint16_t numerologyBwp1 = 1;
    double totalTxPower = 43;

    bool IsImageRenderedEnabled = false;
    Vector CameraPosition(Vector(-150.0, -600.0, 500.0));
    Vector CameraLookAt(Vector(-150.0, 0.0, 0.0));
    std::string filenamePrefix = "sionna-rt-scene-";
    std::string filedirectory = "sionna-rt-images";

    std::string guiSrc;               // empty = GUI bridge disabled
    std::string guiHost = "localhost";

    SionnaRtChannelModel::RtPathSolverConfig RtPathSolverConfig;
    RtPathSolverConfig.maxDepth = 3;
    RtPathSolverConfig.los = true;
    RtPathSolverConfig.specularReflection = true;
    RtPathSolverConfig.diffuseReflection = false;
    RtPathSolverConfig.diffraction = true;
    RtPathSolverConfig.edgeDiffraction = true;
    RtPathSolverConfig.refraction = false;
    RtPathSolverConfig.syntheticArray = false;
    RtPathSolverConfig.seed = 49;

    CommandLine cmd(__FILE__);
    cmd.AddValue("sionnaScene", "Sionna RT scene XML path", sionnaScene);
    cmd.AddValue("gnbPositions", "gNB position CSV path", gnbPositionsPath);
    cmd.AddValue("sumoTrace", "UE trace CSV path", sumoTracePath);
    cmd.AddValue("N_Ues", "Expected number of UEs in the trace", nUes);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.AddValue("trafficStart", "Downlink UDP traffic start time", trafficStart);
    cmd.AddValue("requireTraffic",
                 "Return non-zero status if no downlink bytes reach any UE",
                 requireTraffic);
    cmd.AddValue("sionnaUpdatePeriod", "Sionna RT channel update period", sionnaUpdatePeriod);
    cmd.AddValue("centralFrequencyBand1", "Center frequency in Hz", centralFrequencyBand1);
    cmd.AddValue("bandwidthBand1", "Bandwidth in Hz", bandwidthBand1);
    cmd.AddValue("numerologyBwp1", "Numerology", numerologyBwp1);
    cmd.AddValue("totalTxPower", "Total gNB tx power (dBm)", totalTxPower);
    cmd.AddValue("renderImages", "Enable rendering of scene images to file", IsImageRenderedEnabled);
    cmd.AddValue("cameraPosition", "Camera position for scene rendering", CameraPosition);
    cmd.AddValue("cameraLookAt", "Camera look-at point for scene rendering", CameraLookAt);
    cmd.AddValue("maxDepth", "Maximum reflection/refraction depth", RtPathSolverConfig.maxDepth);
    cmd.AddValue("diffraction", "Enable diffraction", RtPathSolverConfig.diffraction);
    cmd.AddValue("guiSrc",
                 "Path to ns-O-RAN-flexric's contrib/sionna/gui/src (empty disables the GUI bridge)",
                 guiSrc);
    cmd.AddValue("guiHost", "Polyscope GUI ZMQ bridge host", guiHost);
    cmd.Parse(argc, argv);

    NS_ABORT_IF(centralFrequencyBand1 < 0.5e9 && centralFrequencyBand1 > 100e9);

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    py::object guiClient = ConnectGuiZmqBridge(guiSrc, guiHost);

    std::vector<UeTrace> ueTraces = LoadUeTrace(sumoTracePath, nUes);
    std::vector<GnbPosition> gnbPositions = LoadGnbPositions(gnbPositionsPath);

    NodeContainer gnbNodes;
    gnbNodes.Create(gnbPositions.size());
    NodeContainer ueNodes;
    ueNodes.Create(ueTraces.size());

    NS_LOG_UNCOND("Creating " << ueNodes.GetN() << " UEs and " << gnbNodes.GetN() << " gNBs");

    // gNB mobility: fixed positions from CSV.
    Ptr<ListPositionAllocator> gnbPositionAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < gnbPositions.size(); ++i)
    {
        gnbPositionAlloc->Add(gnbPositions[i].position);
        NS_LOG_UNCOND("gNB '" << gnbPositions[i].externalId << "' -> gNB " << i << " pos=("
                              << gnbPositions[i].position.x << "," << gnbPositions[i].position.y
                              << "," << gnbPositions[i].position.z << ")");
        SendGnbPositionToGui(guiClient, gnbPositions[i].externalId, gnbPositions[i].position);
    }
    MobilityHelper gnbMobility;
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.SetPositionAllocator(gnbPositionAlloc);
    gnbMobility.Install(gnbNodes);

    // UE mobility: start at each trace's initial position, then teleport to
    // each subsequent sample's position at that sample's timestamp -- same
    // mechanism as scenario-zero-sionna-kyunghee.cc's SionnaMobilityModel
    // driver, just against the stock ConstantPositionMobilityModel since the
    // Sionna RT channel model reads position off whatever MobilityModel is
    // installed and doesn't need a custom class.
    Ptr<ListPositionAllocator> uePositionAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        uePositionAlloc->Add(InitialPosition(ueTraces[u]));
    }
    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    ueMobility.SetPositionAllocator(uePositionAlloc);
    ueMobility.Install(ueNodes);

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<ConstantPositionMobilityModel> mobility =
            ueNodes.Get(u)->GetObject<ConstantPositionMobilityModel>();
        NS_LOG_UNCOND("UE trace '" << ueTraces[u].externalId << "' -> UE " << u << ", node "
                                   << ueNodes.Get(u)->GetId());
        SendUePositionToGui(guiClient, ueTraces[u].externalId, InitialPosition(ueTraces[u]));
        const std::string ueExternalId = ueTraces[u].externalId;
        for (const auto& sample : ueTraces[u].samples)
        {
            if (sample.time <= 0.0 || Seconds(sample.time) > simTime)
            {
                continue;
            }
            Simulator::Schedule(Seconds(sample.time), [mobility, sample, u, &guiClient, ueExternalId]() {
                mobility->SetPosition(sample.position);
                SendUePositionToGui(guiClient, ueExternalId, sample.position);
                NS_LOG_UNCOND("[trace] t=" << Simulator::Now().GetSeconds() << " UE " << u
                                           << " active=" << sample.active << " pos=("
                                           << sample.position.x << "," << sample.position.y << ","
                                           << sample.position.z << ")");
            });
        }
    }

    // NR / Sionna RT channel setup (same pattern as cttc-nr-demo-sionna-rt.cc).
    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;
    CcBwpCreator::SimpleOperationBandConf bandConf1(centralFrequencyBand1,
                                                    bandwidthBand1,
                                                    numCcPerBand);
    OperationBandInfo band1 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf1);

    double x = pow(10, totalTxPower / 10);
    double totalBandwidth = bandwidthBand1;

    Config::SetDefault("ns3::SionnaRtChannelModel::UpdatePeriod", TimeValue(sionnaUpdatePeriod));
    Config::SetDefault("ns3::SionnaRtChannelModel::Scenario", StringValue(sionnaScene));
    Config::SetDefault("ns3::SionnaRtChannelModel::IsImageRenderedEnabled",
                       BooleanValue(IsImageRenderedEnabled));
    Config::SetDefault("ns3::SionnaRtChannelModel::CameraPosition", VectorValue(CameraPosition));
    Config::SetDefault("ns3::SionnaRtChannelModel::CameraLookAt", VectorValue(CameraLookAt));
    Config::SetDefault("ns3::SionnaRtChannelModel::OutputImageName", StringValue(filenamePrefix));
    Config::SetDefault("ns3::SionnaRtChannelModel::OutputImageDirectory",
                       StringValue(filedirectory));

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->SetAttribute("ChannelModel", StringValue("SionnaRT"));
    channelHelper->ConfigureSpectrumFactory(SionnaRtSpectrumPropagationLossModel::GetTypeId());
    channelHelper->AssignChannelsToBands({band1}, NrChannelHelper::INIT_FADING);
    allBwps = CcBwpCreator::GetAllBwps({band1});
    SetSionnaRtPathSolverConfig(allBwps, RtPathSolverConfig);

    Packet::EnableChecking();
    Packet::EnablePrinting();

    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));
    nrEpcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(0));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(0));

    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice(ueNodes, allBwps);

    int64_t randomStream = 1;
    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    nrHelper->AssignStreams(ueNetDev, randomStream);

    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i)
    {
        NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)
            ->SetAttribute("Numerology", UintegerValue(numerologyBwp1));
        NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)
            ->SetAttribute("TxPower",
                          DoubleValue(10 * log10((bandwidthBand1 / totalBandwidth) * x)));
    }

    auto [remoteHost, remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.000));

    InternetStackHelper internet;
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface = nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueNetDev));
    nrHelper->AttachToClosestGnb(ueNetDev, gnbNetDev);

    // Single downlink UDP flow per UE, matching scenario-zero-sionna-kyunghee.cc,
    // gated by the trace's active/inactive windows.
    uint16_t dlPort = 1234;
    ApplicationContainer serverApps;
    ApplicationContainer clientApps;
    UdpServerHelper dlPacketSink(dlPort);
    serverApps.Add(dlPacketSink.Install(ueNodes));

    NrQosFlow dlFlow(NrQosFlow::NGBR_LOW_LAT_EMBB);
    Ptr<NrQosRule> dlRule = Create<NrQosRule>();
    NrQosRule::PacketFilter dlpf;
    dlpf.localPortStart = dlPort;
    dlpf.localPortEnd = dlPort;
    dlRule->Add(dlpf);

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<NetDevice> ueDevice = ueNetDev.Get(u);
        nrHelper->ActivateDedicatedQosFlow(ueDevice, dlFlow, dlRule);

        auto installClient = [&](double start, double stop) {
            start = std::max(start, trafficStart.GetSeconds());
            stop = std::min(stop, simTime.GetSeconds());
            if (stop <= start)
            {
                return;
            }
            UdpClientHelper dlClient(ueIpIface.GetAddress(u), dlPort);
            dlClient.SetAttribute("Interval", TimeValue(MilliSeconds(20)));
            dlClient.SetAttribute("MaxPackets", UintegerValue(1000000));
            dlClient.SetAttribute("PacketSize", UintegerValue(1024));
            ApplicationContainer clientApp = dlClient.Install(remoteHost);
            clientApp.Start(Seconds(start));
            clientApp.Stop(Seconds(stop));
            clientApps.Add(clientApp);
        };

        bool active = false;
        double activeStart = 0.0;
        for (const auto& sample : ueTraces[u].samples)
        {
            if (sample.time < 0.0)
            {
                active = sample.active;
                if (active)
                {
                    activeStart = 0.0;
                }
                continue;
            }
            if (sample.active && !active)
            {
                active = true;
                activeStart = sample.time;
            }
            else if (!sample.active && active)
            {
                installClient(activeStart, sample.time);
                active = false;
            }
        }
        if (active)
        {
            installClient(activeStart, simTime.GetSeconds());
        }
    }
    serverApps.Start(trafficStart);
    serverApps.Stop(simTime);

    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(ueNodes);
    Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);

    Simulator::Stop(simTime);
    Simulator::Run();

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    double averageFlowThroughput = 0.0;
    double totalDownlinkRxBytes = 0;
    double flowDuration = (simTime - trafficStart).GetSeconds();
    for (auto i = stats.begin(); i != stats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        std::cout << "Flow " << i->first << " (" << t.sourceAddress << ":" << t.sourcePort
                  << " -> " << t.destinationAddress << ":" << t.destinationPort << ")\n";
        std::cout << "  Tx Packets: " << i->second.txPackets << "\n";
        std::cout << "  Rx Bytes:   " << i->second.rxBytes << "\n";
        totalDownlinkRxBytes += i->second.rxBytes;
        if (i->second.rxPackets > 0)
        {
            averageFlowThroughput += i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000;
            std::cout << "  Throughput: " << i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000
                      << " Mbps\n";
        }
        std::cout << "  Rx Packets: " << i->second.rxPackets << "\n";
    }
    std::cout << "\nMean flow throughput: " << (averageFlowThroughput / stats.size()) << " Mbps\n";

    Simulator::Destroy();

    const bool trafficMissing = requireTraffic && totalDownlinkRxBytes == 0;
    return trafficMissing ? 1 : 0;
}
