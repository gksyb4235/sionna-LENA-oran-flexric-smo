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
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

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

// scratch/influx_writer.py is a standalone module (stdlib-only, no
// polyscope/GUI dependency) so it can be imported regardless of whether
// --guiSrc is set.
static py::object
ConnectInfluxWriter(const std::string& influxSrc,
                    const std::string& influxHost,
                    uint16_t influxPort,
                    const std::string& influxDb)
{
    if (influxSrc.empty())
    {
        return py::none();
    }
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, influxSrc);
    py::object client = py::module_::import("influx_writer").attr("InfluxWriter")(
        py::arg("host") = influxHost,
        py::arg("port") = influxPort,
        py::arg("database") = influxDb);
    client.attr("create_database")();
    NS_LOG_UNCOND("[influx] writing KPIs to " << influxHost << ":" << influxPort << "/" << influxDb);
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

// color: (r,g,b) in [0,1], reusing Vector as a cheap RGB triple.
static void
SendColorToGui(py::object& client, const std::string& name, const Vector& color)
{
    if (client.is_none())
    {
        return;
    }
    try
    {
        client.attr("send_set_color")(name, py::make_tuple(color.x, color.y, color.z));
    }
    catch (const py::error_already_set& error)
    {
        NS_LOG_UNCOND("[gui] send_set_color failed (ignoring): " << error.what());
    }
}

// Fixed, high-contrast palette so each gNB (and, through it, whichever UEs
// are currently served by that gNB) gets a visually distinct color in the
// GUI. Cycles if there are more gNBs than colors.
static const std::vector<Vector> kCellColorPalette = {
    Vector(0.90, 0.20, 0.20), // red
    Vector(0.20, 0.45, 0.90), // blue
    Vector(0.20, 0.75, 0.35), // green
    Vector(0.95, 0.60, 0.10), // orange
    Vector(0.65, 0.30, 0.85), // purple
    Vector(0.20, 0.80, 0.80), // teal
};

// ---- Handover event logging ----
//
// NrGnbRrc::HandoverStart/HandoverEndOk fire with just (imsi, cellId, rnti[,
// targetCellId]) -- no position -- so to see *where* the handover happened
// we look the UE node up by IMSI in a small global map filled once at
// mobility setup time and read its live position when the trace fires.
// The same maps also drive the GUI's per-UE color: on every handover, the
// UE's point is recolored to match its new serving gNB's color, so serving
// cell is visible at a glance instead of only in the log.
static std::map<uint64_t, Ptr<Node>> g_imsiToUeNode;
static std::map<uint64_t, std::string> g_imsiToUeName;
static std::map<uint16_t, Vector> g_cellIdToColor;
static py::object* g_guiClient = nullptr; // set once in main(); nullptr if GUI disabled

static void
RecolorUeForCell(uint64_t imsi, uint16_t cellId)
{
    if (!g_guiClient || g_guiClient->is_none())
    {
        return;
    }
    auto nameIt = g_imsiToUeName.find(imsi);
    auto colorIt = g_cellIdToColor.find(cellId);
    if (nameIt == g_imsiToUeName.end() || colorIt == g_cellIdToColor.end())
    {
        return;
    }
    SendColorToGui(*g_guiClient, nameIt->second, colorIt->second);
}

// ---- InfluxDB KPI reporting ----
//
// Feeds the "ue_kpi"/"cell_kpi" measurements read by monitoring/grafana's
// dashboards (see scratch/influx_writer.py for the schema). Kept entirely
// separate from the GUI/E2 code paths above -- this just adds more
// Config::Connect listeners on the same trace sources plus a periodic
// poll, none of which touch the channel/PHY computation.
static py::object* g_influxClient = nullptr; // nullptr if disabled

struct UeKpiState
{
    std::string name;
    std::map<uint16_t, double> rsrpDbm; //!< last RSRP per cellId (serving + neighbours)
    Ptr<UdpServer> serverApp;
    uint64_t lastRxPackets = 0;
    uint32_t hoCount = 0;
    double lastHoTime = -1e9;
    uint16_t lastHoSource = 0;
    uint16_t lastHoTarget = 0;
    bool isPingPong = false;
    double dlSinrDb = 0.0;
    uint8_t dlMcs = 0;
};

static std::map<uint64_t, UeKpiState> g_ueKpi; //!< keyed by IMSI

struct CellKpiState
{
    std::string name;
    double txPowerDbm = 0;
    Ptr<NetDevice> gnbDev; //!< to read the live RET tilt/bearing at report time
    uint32_t hoInCount = 0;
    uint32_t hoOutCount = 0;
    uint32_t pingPongCount = 0;
    uint64_t prbUsedAccum = 0;     //!< sum of usedReg since the last report, reset each report
    uint64_t prbCapacityAccum = 0; //!< sum of availableRb*availableSym since the last report
};

static std::map<uint16_t, CellKpiState> g_cellKpi; //!< keyed by cellId

// rnti is only unique within a single cell, not simulation-wide; keyed on
// (cellId, rnti) where the trace provides a cellId, and on rnti alone (best
// effort, most recently seen owner wins) where it doesn't (CqiFeedbackTrace).
static std::map<std::pair<uint16_t, uint16_t>, uint64_t> g_cellRntiToImsi;
static std::map<uint16_t, uint64_t> g_rntiToImsi;

static std::map<uint64_t, uint16_t> g_pendingHoSourceCell; //!< IMSI -> source cell of an in-flight HO
static double g_handoverTtTMs = 0.0;
static double g_handoverHysteresisDb = 0.0;
static constexpr double kPingPongWindowSec = 10.0;

static void
RecordRsrpForKpi(uint64_t imsi, uint16_t cellId, uint16_t rnti, NrRrcSap::MeasurementReport report)
{
    g_cellRntiToImsi[{cellId, rnti}] = imsi;
    g_rntiToImsi[rnti] = imsi;

    auto it = g_ueKpi.find(imsi);
    if (it == g_ueKpi.end())
    {
        return;
    }
    it->second.rsrpDbm[cellId] =
        static_cast<double>(report.measResults.measResultPCell.rsrpResult) - 140.0;
    if (report.measResults.haveMeasResultNeighCells)
    {
        for (const auto& neigh : report.measResults.measResultListEutra)
        {
            if (neigh.haveRsrpResult)
            {
                it->second.rsrpDbm[neigh.physCellId] =
                    static_cast<double>(neigh.rsrpResult) - 140.0;
            }
        }
    }
}

static void
RecordDlSinrForKpi(uint16_t cellId, uint16_t rnti, double sinrLinear, uint16_t /* bwpId */)
{
    auto rntiIt = g_cellRntiToImsi.find({cellId, rnti});
    if (rntiIt == g_cellRntiToImsi.end())
    {
        return;
    }
    auto ueIt = g_ueKpi.find(rntiIt->second);
    if (ueIt == g_ueKpi.end())
    {
        return;
    }
    // NrUePhy::ComputeAvgSinr() returns a linear ratio (confirmed by
    // NrUePhy::m_rlfDetectionEvent computing 10*log10(ComputeAvgSinr(...))
    // for the same quantity), so convert to dB here.
    ueIt->second.dlSinrDb = (sinrLinear > 0.0) ? 10.0 * std::log10(sinrLinear) : -30.0;
}

static void
RecordCqiFeedbackForKpi(uint16_t rnti, uint8_t /* cqi */, uint8_t mcs, uint8_t /* ri */)
{
    auto rntiIt = g_rntiToImsi.find(rnti);
    if (rntiIt == g_rntiToImsi.end())
    {
        return;
    }
    auto ueIt = g_ueKpi.find(rntiIt->second);
    if (ueIt == g_ueKpi.end())
    {
        return;
    }
    ueIt->second.dlMcs = mcs;
}

static void
RecordSlotDataStatsForKpi(const SfnSf& /* sfnSf */,
                          uint32_t /* scheduledUe */,
                          uint32_t usedReg,
                          uint32_t /* usedSym */,
                          uint32_t availableRb,
                          uint32_t availableSym,
                          uint16_t /* bwpId */,
                          uint16_t cellId)
{
    auto it = g_cellKpi.find(cellId);
    if (it == g_cellKpi.end())
    {
        return;
    }
    it->second.prbUsedAccum += usedReg;
    it->second.prbCapacityAccum += static_cast<uint64_t>(availableRb) * availableSym;
}

static void
ReportKpiToInflux(NetDeviceContainer ueNetDev, double intervalSec)
{
    if (!g_influxClient || g_influxClient->is_none())
    {
        return;
    }

    std::map<uint16_t, uint32_t> cellUeCount;
    std::map<uint16_t, double> cellRsrpSum;
    std::map<uint16_t, double> cellThroughputSum;

    for (uint32_t i = 0; i < ueNetDev.GetN(); ++i)
    {
        auto ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(i));
        uint64_t imsi = ueDev->GetImsi();
        auto it = g_ueKpi.find(imsi);
        if (it == g_ueKpi.end())
        {
            continue;
        }
        UeKpiState& ue = it->second;
        uint16_t servingCellId = ueDev->GetCellId();

        double throughputMbps = 0.0;
        if (ue.serverApp)
        {
            uint64_t rx = ue.serverApp->GetReceived();
            uint64_t deltaPackets = (rx >= ue.lastRxPackets) ? (rx - ue.lastRxPackets) : 0;
            ue.lastRxPackets = rx;
            // PacketSize is fixed at 1024 bytes on the UdpClient below.
            throughputMbps = (deltaPackets * 1024.0 * 8.0) / (intervalSec * 1e6);
        }

        double rsrpServing = 0.0;
        if (auto rIt = ue.rsrpDbm.find(servingCellId); rIt != ue.rsrpDbm.end())
        {
            rsrpServing = rIt->second;
        }

        std::string bestNeighborName;
        double bestNeighborRsrp = -140.0;
        for (const auto& cellRsrp : ue.rsrpDbm)
        {
            if (cellRsrp.first == servingCellId)
            {
                continue;
            }
            if (cellRsrp.second > bestNeighborRsrp)
            {
                bestNeighborRsrp = cellRsrp.second;
                bestNeighborName.clear();
                if (auto cnIt = g_cellKpi.find(cellRsrp.first); cnIt != g_cellKpi.end())
                {
                    bestNeighborName = cnIt->second.name;
                }
            }
        }

        std::string servingCellName;
        if (auto cIt = g_cellKpi.find(servingCellId); cIt != g_cellKpi.end())
        {
            servingCellName = cIt->second.name;
        }

        double secondsSinceHo =
            (ue.hoCount > 0) ? (Simulator::Now().GetSeconds() - ue.lastHoTime) : -1.0;

        try
        {
            g_influxClient->attr("write_ue_kpi")(ue.name,
                                                 servingCellName,
                                                 rsrpServing,
                                                 bestNeighborName,
                                                 bestNeighborRsrp,
                                                 throughputMbps,
                                                 ue.hoCount,
                                                 secondsSinceHo,
                                                 ue.isPingPong,
                                                 ue.dlSinrDb,
                                                 static_cast<int>(ue.dlMcs));
        }
        catch (const py::error_already_set& error)
        {
            NS_LOG_UNCOND("[influx] write_ue_kpi failed (ignoring): " << error.what());
        }

        cellUeCount[servingCellId]++;
        cellRsrpSum[servingCellId] += rsrpServing;
        cellThroughputSum[servingCellId] += throughputMbps;
    }

    for (auto& cellEntry : g_cellKpi)
    {
        uint16_t cellId = cellEntry.first;
        CellKpiState& cell = cellEntry.second;
        uint32_t numUes = cellUeCount.count(cellId) ? cellUeCount[cellId] : 0;
        double avgRsrp = (numUes > 0) ? (cellRsrpSum[cellId] / numUes) : 0.0;
        double aggThroughput = cellThroughputSum.count(cellId) ? cellThroughputSum[cellId] : 0.0;

        double retTiltDeg = 0.0;
        double retBearingDeg = 0.0;
        if (cell.gnbDev)
        {
            Ptr<UniformPlanarArray> antenna = DynamicCast<UniformPlanarArray>(
                NrHelper::GetGnbPhy(cell.gnbDev, 0)->GetSpectrumPhy()->GetAntenna());
            if (antenna)
            {
                DoubleValue tiltVal;
                antenna->GetAttribute("DowntiltAngle", tiltVal);
                retTiltDeg = tiltVal.Get() * 180.0 / M_PI;
                DoubleValue bearingVal;
                antenna->GetAttribute("BearingAngle", bearingVal);
                retBearingDeg = bearingVal.Get() * 180.0 / M_PI;
            }
        }

        double prbUtilizationPct = (cell.prbCapacityAccum > 0)
                                       ? (100.0 * static_cast<double>(cell.prbUsedAccum) /
                                          static_cast<double>(cell.prbCapacityAccum))
                                       : 0.0;
        cell.prbUsedAccum = 0;
        cell.prbCapacityAccum = 0;

        try
        {
            g_influxClient->attr("write_cell_kpi")(cell.name,
                                                   cell.txPowerDbm,
                                                   retTiltDeg,
                                                   retBearingDeg,
                                                   g_handoverTtTMs,
                                                   g_handoverHysteresisDb,
                                                   static_cast<int>(numUes),
                                                   avgRsrp,
                                                   cell.hoInCount,
                                                   cell.hoOutCount,
                                                   cell.pingPongCount,
                                                   aggThroughput,
                                                   prbUtilizationPct);
        }
        catch (const py::error_already_set& error)
        {
            NS_LOG_UNCOND("[influx] write_cell_kpi failed (ignoring): " << error.what());
        }
    }

    Simulator::Schedule(Seconds(intervalSec), &ReportKpiToInflux, ueNetDev, intervalSec);
}

static void
LogHandoverStart(uint64_t imsi, uint16_t sourceCellId, uint16_t rnti, uint16_t targetCellId)
{
    g_pendingHoSourceCell[imsi] = sourceCellId;
    Vector pos(0, 0, 0);
    if (auto it = g_imsiToUeNode.find(imsi); it != g_imsiToUeNode.end())
    {
        pos = it->second->GetObject<MobilityModel>()->GetPosition();
    }
    NS_LOG_UNCOND("[HO] t=" << Simulator::Now().GetSeconds() << "s START imsi=" << imsi
                            << " rnti=" << rnti << " " << sourceCellId << " -> " << targetCellId
                            << " ue_pos=(" << pos.x << "," << pos.y << "," << pos.z << ")");
}

static void
LogHandoverEndOk(uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    RecolorUeForCell(imsi, cellId);

    uint16_t sourceCellId = 0;
    if (auto it = g_pendingHoSourceCell.find(imsi); it != g_pendingHoSourceCell.end())
    {
        sourceCellId = it->second;
        g_pendingHoSourceCell.erase(it);
    }
    if (auto ueIt = g_ueKpi.find(imsi); ueIt != g_ueKpi.end())
    {
        UeKpiState& ue = ueIt->second;
        double now = Simulator::Now().GetSeconds();
        // Ping-pong: this handover reverses the immediately preceding one
        // (B->A right after A->B) within a short window.
        ue.isPingPong = (ue.hoCount > 0) && (sourceCellId == ue.lastHoTarget) &&
                        (cellId == ue.lastHoSource) && (now - ue.lastHoTime) < kPingPongWindowSec;
        ue.hoCount++;
        ue.lastHoTime = now;
        ue.lastHoSource = sourceCellId;
        ue.lastHoTarget = cellId;

        if (auto srcCellIt = g_cellKpi.find(sourceCellId); srcCellIt != g_cellKpi.end())
        {
            srcCellIt->second.hoOutCount++;
            if (ue.isPingPong)
            {
                srcCellIt->second.pingPongCount++;
            }
        }
        if (auto dstCellIt = g_cellKpi.find(cellId); dstCellIt != g_cellKpi.end())
        {
            dstCellIt->second.hoInCount++;
        }
    }

    Vector pos(0, 0, 0);
    if (auto it = g_imsiToUeNode.find(imsi); it != g_imsiToUeNode.end())
    {
        pos = it->second->GetObject<MobilityModel>()->GetPosition();
    }
    NS_LOG_UNCOND("[HO] t=" << Simulator::Now().GetSeconds() << "s END_OK imsi=" << imsi
                            << " rnti=" << rnti << " now_serving_cell=" << cellId << " ue_pos=("
                            << pos.x << "," << pos.y << "," << pos.z << ")");
}

int
main(int argc, char* argv[])
{
    py::scoped_interpreter guard{};

    // Paths are relative to the repository root. Run the binary from there, or
    // override them through the corresponding command-line arguments.
    std::string sionnaScene = "scenes/khu-real/KHU_Cropped_Sionna_RT.xml";
    std::string gnbPositionsPath = "scenarios/khu-real/gnbs-ret.csv";
    std::string sumoTracePath = "scenarios/khu-real/ues-15.csv";
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

    std::string influxSrc;            // empty = InfluxDB KPI reporting disabled
    std::string influxHost = "localhost";
    uint16_t influxPort = 8086;
    std::string influxDb = "nr_kpi";
    double kpiReportInterval = 1.0;

    // Applied explicitly here (rather than left to --ns3::NrA3RsrpHandoverAlgorithm::*)
    // so the cell_kpi dashboard always shows the value actually in effect.
    double handoverTtTMs = 256.0;
    double handoverHysteresisDb = 3.0;

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
    cmd.AddValue("influxSrc",
                 "Path to the directory containing influx_writer.py (empty disables "
                 "InfluxDB KPI reporting)",
                 influxSrc);
    cmd.AddValue("influxHost", "InfluxDB host", influxHost);
    cmd.AddValue("influxPort", "InfluxDB HTTP port", influxPort);
    cmd.AddValue("influxDb", "InfluxDB database name", influxDb);
    cmd.AddValue("kpiReportInterval", "Seconds between KPI reports to InfluxDB", kpiReportInterval);
    cmd.AddValue("handoverTtt", "Handover time-to-trigger (ms)", handoverTtTMs);
    cmd.AddValue("handoverHysteresis", "Handover hysteresis (dB)", handoverHysteresisDb);
    cmd.Parse(argc, argv);

    g_handoverTtTMs = handoverTtTMs;
    g_handoverHysteresisDb = handoverHysteresisDb;

    NS_ABORT_IF(centralFrequencyBand1 < 0.5e9 && centralFrequencyBand1 > 100e9);

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    py::object guiClient = ConnectGuiZmqBridge(guiSrc, guiHost);
    g_guiClient = &guiClient;

    py::object influxClient = ConnectInfluxWriter(influxSrc, influxHost, influxPort, influxDb);
    g_influxClient = &influxClient;

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

        // NR assigns cellId = (gNB index * componentCarriers) + ccIndex + 1;
        // with one CC per gNB here, that's simply i+1. Record the color now
        // so UEs can be recolored to match as soon as they attach/handover.
        const uint16_t cellId = static_cast<uint16_t>(i + 1);
        const Vector cellColor = kCellColorPalette[i % kCellColorPalette.size()];
        g_cellIdToColor[cellId] = cellColor;
        SendColorToGui(guiClient, gnbPositions[i].externalId, cellColor);
        g_cellKpi[cellId].name = gnbPositions[i].externalId;
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

    // The handover algorithm object is created as part of InstallGnbDevice(),
    // so its attributes must be set on the factory before installing gNBs.
    // Keep these command-line values and the KPI labels tied to the same
    // effective configuration.
    if (nrHelper->GetHandoverAlgorithmType() == "ns3::NrA3RsrpHandoverAlgorithm")
    {
        nrHelper->SetHandoverAlgorithmAttribute("TimeToTrigger",
                                                TimeValue(MilliSeconds(handoverTtTMs)));
        nrHelper->SetHandoverAlgorithmAttribute("Hysteresis",
                                                DoubleValue(handoverHysteresisDb));
    }

    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice(ueNodes, allBwps);

    // X2 carries the actual handover signaling (HANDOVER REQUEST/ACK/etc.)
    // between gNBs. Without this, a handover algorithm that decides to
    // trigger a handover (e.g. --ns3::NrHelper::HandoverAlgorithm=
    // ns3::NrA3RsrpHandoverAlgorithm) crashes on the first attempt
    // (NrEpcX2::DoSendHandoverRequest asserts on a missing X2 socket).
    if (gnbNodes.GetN() > 1)
    {
        nrHelper->AddX2Interface(gnbNodes);
    }

    // Handover event logging: map IMSI -> UE node/external-id so the trace
    // callbacks below can report where each handover happened and recolor
    // the right GUI point.
    for (uint32_t i = 0; i < ueNetDev.GetN(); ++i)
    {
        auto ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(i));
        g_imsiToUeNode[ueDev->GetImsi()] = ueNodes.Get(i);
        g_imsiToUeName[ueDev->GetImsi()] = ueTraces[i].externalId;
        g_ueKpi[ueDev->GetImsi()].name = ueTraces[i].externalId;
    }
    Config::ConnectWithoutContextFailSafe("/NodeList/*/DeviceList/*/NrGnbRrc/HandoverStart",
                                          MakeCallback(&LogHandoverStart));
    Config::ConnectWithoutContextFailSafe("/NodeList/*/DeviceList/*/NrGnbRrc/HandoverEndOk",
                                          MakeCallback(&LogHandoverEndOk));
    Config::ConnectWithoutContextFailSafe("/NodeList/*/DeviceList/*/NrGnbRrc/RecvMeasurementReport",
                                          MakeCallback(&RecordRsrpForKpi));

    bool sinrConnected = Config::ConnectWithoutContextFailSafe(
        "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/NrUePhy/DlDataSinr",
        MakeCallback(&RecordDlSinrForKpi));
    bool cqiConnected = Config::ConnectWithoutContextFailSafe(
        "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/NrUePhy/CqiFeedbackTrace",
        MakeCallback(&RecordCqiFeedbackForKpi));
    bool slotStatsConnected = Config::ConnectWithoutContextFailSafe(
        "/NodeList/*/DeviceList/*/BandwidthPartMap/*/NrGnbPhy/SlotDataStats",
        MakeCallback(&RecordSlotDataStatsForKpi));
    NS_LOG_UNCOND("[kpi] trace connect: DlDataSinr=" << sinrConnected
                                                     << " CqiFeedbackTrace=" << cqiConnected
                                                     << " SlotDataStats=" << slotStatsConnected);

    int64_t randomStream = 1;
    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    nrHelper->AssignStreams(ueNetDev, randomStream);

    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i)
    {
        NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)
            ->SetAttribute("Numerology", UintegerValue(numerologyBwp1));
        const double txPowerDbm = 10 * log10((bandwidthBand1 / totalBandwidth) * x);
        NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)->SetAttribute("TxPower", DoubleValue(txPowerDbm));

        const uint16_t cellId = static_cast<uint16_t>(i + 1);
        g_cellKpi[cellId].gnbDev = gnbNetDev.Get(i);
        g_cellKpi[cellId].txPowerDbm = txPowerDbm;
    }

    auto [remoteHost, remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.000));

    InternetStackHelper internet;
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface = nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueNetDev));
    nrHelper->AttachToClosestGnb(ueNetDev, gnbNetDev);

    // Color each UE to match its initial serving cell in the GUI. Attach is
    // via ideal RRC signaling and isn't instantaneous, so this is scheduled
    // for trafficStart rather than read synchronously here.
    Simulator::Schedule(trafficStart, [ueNetDev]() {
        for (uint32_t i = 0; i < ueNetDev.GetN(); ++i)
        {
            auto ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(i));
            RecolorUeForCell(ueDev->GetImsi(), ueDev->GetCellId());
        }
    });

    // Single downlink UDP flow per UE, matching scenario-zero-sionna-kyunghee.cc,
    // gated by the trace's active/inactive windows.
    uint16_t dlPort = 1234;
    ApplicationContainer serverApps;
    ApplicationContainer clientApps;
    UdpServerHelper dlPacketSink(dlPort);
    serverApps.Add(dlPacketSink.Install(ueNodes));
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        auto ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(u));
        g_ueKpi[ueDev->GetImsi()].serverApp = DynamicCast<UdpServer>(serverApps.Get(u));
    }

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

    if (!influxClient.is_none())
    {
        Simulator::Schedule(trafficStart, &ReportKpiToInflux, ueNetDev, kpiReportInterval);
    }

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
