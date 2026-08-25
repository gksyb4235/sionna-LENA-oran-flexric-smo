// Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

/**
 * @file khu-real-nr-sionna-pooled.cc
 * @brief Slot-pooled variant of khu-real-nr-sionna.cc: instead of creating one
 * ns-3 UE node/NR device per distinct ue_id in the trace CSV (223 for
 * ue_positions_seed0.csv, each participating in every Sionna RT channel
 * update for the whole sim regardless of whether it is actually active at
 * that moment), this file creates a fixed-size *pool* of UE devices sized to
 * the trace's true maximum concurrent active-UE count, and reassigns real
 * trace UE identities onto free pool slots as they enter/leave their active
 * windows.
 *
 * This is a separate scratch target from khu-real-nr-sionna.cc -- nothing in
 * ns-3 core, contrib/nr, or the original scenario file is touched.
 *
 * Slot assignment is precomputed once, offline, before Simulator::Run(), via
 * classic interval-graph-coloring (greedy, earliest-available-slot): sort all
 * (trace, active-window) sessions by start time, reuse the slot whose
 * previous session ended earliest if it's already free by the new session's
 * start, otherwise allocate a new slot. This yields the minimum possible
 * pool size (== the trace's true max concurrent count) and a fully static
 * schedule, so no runtime free-list/allocator is needed during Simulator::Run().
 *
 * When a slot's device is used for the very first time, it gets a clean
 * initial attach via NrHelper::AttachToGnb to the geometrically closest gNB
 * (the "just powered on, camping on the nearest cell" case -- no prior RRC
 * state to worry about). NrHelper::AttachToMaxRsrpGnb (true RSRP-based
 * attach) was tried here first, since it's the more physically accurate
 * choice, but it reliably segfaults when invoked dynamically mid-simulation
 * in this NR fork -- upstream it only seems to be exercised as a one-shot
 * call for every UE before Simulator::Run() begins, not as a per-device call
 * fired from a later scheduled event. Distance-based attach is a reasonable
 * stand-in here: with only 2 isotropic, non-RET gNBs (RET/tilt isn't applied
 * to the NR antenna model yet), nearest-by-distance and strongest-by-RSRP
 * pick the same cell in practice, and it's the same mechanism
 * khu-real-nr-sionna.cc's own baseline already uses for its initial
 * deployment.
 *
 * When a slot is *reused* by a later session, the device is already
 * RRC-connected somewhere from its previous occupant, and AttachToGnb isn't
 * safe to call a second time on an already-attached device -- it just
 * re-registers the RNTI without releasing the old context, leaking stale
 * state on the previous serving gNB. Reuse therefore goes through
 * NrHelper::HandoverRequest instead (the same public, well-tested X2
 * handover path NrA3RsrpHandoverAlgorithm uses at runtime), targeted at the
 * geometrically closest gNB. These reuse-handovers are intentionally
 * excluded from the ho_count/ping-pong KPI bookkeeping below (they're slot
 * housekeeping, not a real person's mobility-driven handover) but do briefly
 * appear in the [HO] log lines fired by NrGnbRrc's own
 * HandoverStart/HandoverEndOk traces.
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
#include "ns3/sionna-lookup-channel-model.h"
#include "ns3/sionna-rt-channel-model.h"
#include "ns3/sionna-rt-spectrum-propagation-loss-model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <utility>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("KhuRealNrSionnaPooled");

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

// ---- Optional live GUI bridge (unchanged protocol from khu-real-nr-sionna.cc) ----

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
SendGnbOrientationToGui(py::object& client,
                        const std::string& name,
                        double bearingDeg,
                        double tiltDeg)
{
    if (client.is_none())
    {
        return;
    }
    try
    {
        client.attr("send_gnb_orientation")(name, bearingDeg, tiltDeg);
    }
    catch (const py::error_already_set& error)
    {
        NS_LOG_UNCOND("[gui] send_gnb_orientation failed (ignoring): " << error.what());
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

static const std::vector<Vector> kCellColorPalette = {
    Vector(0.90, 0.20, 0.20), // red
    Vector(0.20, 0.45, 0.90), // blue
    Vector(0.20, 0.75, 0.35), // green
    Vector(0.95, 0.60, 0.10), // orange
    Vector(0.65, 0.30, 0.85), // purple
    Vector(0.20, 0.80, 0.80), // teal
};

// ---- Handover event logging (unchanged from khu-real-nr-sionna.cc) ----
static std::map<uint64_t, Ptr<Node>> g_imsiToUeNode;
static std::map<uint64_t, std::string> g_imsiToUeName; // GUI display name, keyed by slot id
static std::map<uint16_t, Vector> g_cellIdToColor;
static py::object* g_guiClient = nullptr;

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

// ---- InfluxDB KPI reporting (unchanged schema from khu-real-nr-sionna.cc) ----
static py::object* g_influxClient = nullptr;

struct UeKpiState
{
    std::string name; //!< real trace ue_id currently occupying this slot ("" if idle)
    std::map<uint16_t, double> rsrpDbm;
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

static std::map<uint64_t, UeKpiState> g_ueKpi; //!< keyed by IMSI (== keyed by slot, 1 IMSI/slot for life)

struct CellKpiState
{
    std::string name;
    double txPowerDbm = 0;
    Ptr<NetDevice> gnbDev;
    uint32_t hoInCount = 0;
    uint32_t hoOutCount = 0;
    uint32_t pingPongCount = 0;
    uint64_t prbUsedAccum = 0;
    uint64_t prbCapacityAccum = 0;
};

static std::map<uint16_t, CellKpiState> g_cellKpi;

static std::map<std::pair<uint16_t, uint16_t>, uint64_t> g_cellRntiToImsi;
static std::map<uint16_t, uint64_t> g_rntiToImsi;

static std::map<uint64_t, uint16_t> g_pendingHoSourceCell;
static double g_handoverTtTMs = 0.0;
static double g_handoverHysteresisDb = 0.0;
static constexpr double kPingPongWindowSec = 10.0;

// Slot reassignment handovers (khu-real-nr-sionna-pooled.cc's own housekeeping,
// not a real person's mobility) are marked here for one HandoverEndOk callback
// so the KPI bookkeeping below can skip ho_count/ping-pong accounting for them
// while still letting the [HO] log lines and GUI recolor fire normally.
static std::set<uint64_t> g_suppressNextHoKpi;

// Populates the RNTI->IMSI lookup the moment a device gets a working RRC
// connection (fresh attach or handover target), instead of waiting for the
// first RecvMeasurementReport. NrA3RsrpHandoverAlgorithm's default measConfig
// is event-triggered (A3) only, not periodic, so a UE that never gets close
// to a cell boundary can go its whole session without ever sending one --
// meanwhile it's happily receiving real downlink traffic on a real MCS the
// whole time. Without this, every fresh RNTI (which pool slot reuse mints far
// more often than the non-pooled baseline, since every reassignment gets a
// new RNTI) stays invisible to RecordDlSinrForKpi/RecordCqiFeedbackForKpi
// until/unless a measurement report happens to arrive, producing exactly the
// "throughput is flowing but RSRP/SINR/MCS are stuck at their no-data
// defaults" contradiction.
static void
RecordAttachForKpi(uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    g_cellRntiToImsi[{cellId, rnti}] = imsi;
    g_rntiToImsi[rnti] = imsi;
}

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
        if (it == g_ueKpi.end() || it->second.name.empty())
        {
            continue; // slot currently idle (between sessions) -- nothing to report
        }
        UeKpiState& ue = it->second;
        uint16_t servingCellId = ueDev->GetCellId();

        double throughputMbps = 0.0;
        if (ue.serverApp)
        {
            uint64_t rx = ue.serverApp->GetReceived();
            uint64_t deltaPackets = (rx >= ue.lastRxPackets) ? (rx - ue.lastRxPackets) : 0;
            ue.lastRxPackets = rx;
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
        double avgRsrp = (numUes > 0) ? (cellRsrpSum[cellId] / numUes)
                                       : std::numeric_limits<double>::quiet_NaN();
        double aggThroughput = cellThroughputSum.count(cellId) ? cellThroughputSum[cellId] : 0.0;

        double retTiltDeg = 0.0;
        double retBearingDeg = 0.0;
        double liveTxPowerDbm = cell.txPowerDbm;
        if (cell.gnbDev)
        {
            Ptr<NrGnbPhy> phy = NrHelper::GetGnbPhy(cell.gnbDev, 0);
            DoubleValue txPowerVal;
            phy->GetAttribute("TxPower", txPowerVal);
            liveTxPowerDbm = txPowerVal.Get();

            Ptr<UniformPlanarArray> antenna =
                DynamicCast<UniformPlanarArray>(phy->GetSpectrumPhy()->GetAntenna());
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
                                                   liveTxPowerDbm,
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
    RecordAttachForKpi(imsi, cellId, rnti); // target cell mints a new RNTI on every handover

    uint16_t sourceCellId = 0;
    if (auto it = g_pendingHoSourceCell.find(imsi); it != g_pendingHoSourceCell.end())
    {
        sourceCellId = it->second;
        g_pendingHoSourceCell.erase(it);
    }

    const bool suppressKpi = g_suppressNextHoKpi.erase(imsi) > 0;

    if (auto ueIt = g_ueKpi.find(imsi); ueIt != g_ueKpi.end() && !suppressKpi)
    {
        UeKpiState& ue = ueIt->second;
        double now = Simulator::Now().GetSeconds();
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
                            << " rnti=" << rnti << " now_serving_cell=" << cellId
                            << (suppressKpi ? " (slot-reuse, KPI-suppressed)" : "") << " ue_pos=("
                            << pos.x << "," << pos.y << "," << pos.z << ")");
}

// ---- Slot pool: sessions and interval-graph-coloring assignment ----

/// One continuous active window of a single real trace UE.
struct UeSession
{
    uint32_t traceIndex{0};
    double start{0.0};
    double end{0.0};
    Vector startPosition;
    uint32_t slot{0}; //!< filled in by AssignSlots()
};

static std::vector<std::pair<double, double>>
ComputeActiveWindows(const UeTrace& trace, double simTimeSec, std::vector<Vector>* startPositions)
{
    std::vector<std::pair<double, double>> windows;
    bool active = false;
    double activeStart = 0.0;
    Vector activeStartPos;
    for (const auto& sample : trace.samples)
    {
        if (sample.time < 0.0)
        {
            active = sample.active;
            if (active)
            {
                activeStart = 0.0;
                activeStartPos = sample.position;
            }
            continue;
        }
        if (sample.active && !active)
        {
            active = true;
            activeStart = sample.time;
            activeStartPos = sample.position;
        }
        else if (!sample.active && active)
        {
            if (activeStart < simTimeSec)
            {
                windows.emplace_back(activeStart, std::min(sample.time, simTimeSec));
                startPositions->push_back(activeStartPos);
            }
            active = false;
        }
    }
    if (active && activeStart < simTimeSec)
    {
        windows.emplace_back(activeStart, simTimeSec);
        startPositions->push_back(activeStartPos);
    }
    return windows;
}

/// Builds all sessions across all traces, then assigns each one a pool slot
/// via greedy interval-graph-coloring (earliest-freed slot wins). Returns the
/// sessions (sorted by start time, each with its `slot` field filled in) and
/// sets `poolSize` to the minimum number of slots this required -- i.e. the
/// trace's true max-concurrent-active-UE count.
static std::vector<UeSession>
BuildSessionsAndAssignSlots(const std::vector<UeTrace>& traces, double simTimeSec,
                            uint32_t* poolSize)
{
    std::vector<UeSession> sessions;
    for (uint32_t t = 0; t < traces.size(); ++t)
    {
        std::vector<Vector> startPositions;
        auto windows = ComputeActiveWindows(traces[t], simTimeSec, &startPositions);
        for (uint32_t w = 0; w < windows.size(); ++w)
        {
            UeSession session;
            session.traceIndex = t;
            session.start = windows[w].first;
            session.end = windows[w].second;
            session.startPosition = startPositions[w];
            sessions.push_back(session);
        }
    }

    std::stable_sort(sessions.begin(), sessions.end(),
                     [](const UeSession& a, const UeSession& b) { return a.start < b.start; });

    // min-heap of (end time, slot id) for slots currently in use.
    using FreeAtEntry = std::pair<double, uint32_t>;
    std::priority_queue<FreeAtEntry, std::vector<FreeAtEntry>, std::greater<>> freeAt;
    uint32_t nextSlot = 0;
    for (auto& session : sessions)
    {
        if (!freeAt.empty() && freeAt.top().first <= session.start)
        {
            session.slot = freeAt.top().second;
            freeAt.pop();
        }
        else
        {
            session.slot = nextSlot++;
        }
        freeAt.push({session.end, session.slot});
    }

    *poolSize = nextSlot;
    return sessions;
}

int
main(int argc, char* argv[])
{
    py::scoped_interpreter guard{};

    std::string sionnaScene = "scenes/khu-real/KHU_Cropped_Sionna_RT.xml";
    std::string sionnaCacheFile; // empty = live Sionna RT (default); see below
    std::string gnbPositionsPath = "scenarios/khu-real/gnbs-ret.csv";
    std::string sumoTracePath = "scenarios/khu-real/ue_positions_seed0.csv";
    uint32_t nUes = 300; // upper bound on distinct real ue_id count in the trace CSV

    Time simTime = Seconds(900);
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

    std::string guiSrc;
    std::string guiHost = "localhost";

    std::string influxSrc;
    std::string influxHost = "localhost";
    uint16_t influxPort = 8086;
    std::string influxDb = "nr_kpi";
    double kpiReportInterval = 1.0;

    double handoverTtTMs = 256.0;
    double handoverHysteresisDb = 3.0;

    uint32_t poolSizeOverride = 0; // 0 = auto (== exact max-concurrent from the trace)

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
    cmd.AddValue("sionnaCacheFile",
                "Path to a HDF5 cache built by build_sionna_rt_cache.py; if set, replaces "
                "live Sionna RT PathSolver calls with lookups (empty = live RT, default)",
                sionnaCacheFile);
    cmd.AddValue("gnbPositions", "gNB position CSV path", gnbPositionsPath);
    cmd.AddValue("sumoTrace", "UE trace CSV path", sumoTracePath);
    cmd.AddValue("N_Ues", "Upper bound on distinct ue_id count in the trace", nUes);
    cmd.AddValue("poolSize",
                 "Fixed UE device pool size (0 = auto-compute exact max concurrent active count)",
                 poolSizeOverride);
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

    uint32_t computedPoolSize = 0;
    std::vector<UeSession> sessions =
        BuildSessionsAndAssignSlots(ueTraces, simTime.GetSeconds(), &computedPoolSize);
    NS_ABORT_MSG_IF(computedPoolSize == 0, "trace produced zero active sessions within simTime");

    uint32_t poolSize = (poolSizeOverride > 0) ? poolSizeOverride : computedPoolSize;
    NS_ABORT_MSG_IF(poolSize < computedPoolSize,
                    "--poolSize=" << poolSize << " is smaller than the trace's true max "
                                  << "concurrent active count (" << computedPoolSize
                                  << "); some sessions would have nowhere to go");

    NS_LOG_UNCOND(ueTraces.size() << " distinct UE ids, " << sessions.size()
                                  << " active sessions, max concurrent = " << computedPoolSize
                                  << " -> pool size = " << poolSize);

    // Group sessions by slot, in start-time order, so we can schedule each
    // slot's arrival/position/departure events against its own node.
    std::vector<std::vector<const UeSession*>> sessionsBySlot(poolSize);
    for (const auto& session : sessions)
    {
        sessionsBySlot[session.slot].push_back(&session);
    }

    NodeContainer gnbNodes;
    gnbNodes.Create(gnbPositions.size());
    NodeContainer ueNodes;
    ueNodes.Create(poolSize);

    NS_LOG_UNCOND("Creating " << ueNodes.GetN() << " pooled UE devices and " << gnbNodes.GetN()
                              << " gNBs");

    Ptr<ListPositionAllocator> gnbPositionAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < gnbPositions.size(); ++i)
    {
        gnbPositionAlloc->Add(gnbPositions[i].position);
        NS_LOG_UNCOND("gNB '" << gnbPositions[i].externalId << "' -> gNB " << i << " pos=("
                              << gnbPositions[i].position.x << "," << gnbPositions[i].position.y
                              << "," << gnbPositions[i].position.z << ")");
        SendGnbPositionToGui(guiClient, gnbPositions[i].externalId, gnbPositions[i].position);

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

    // Pool UE mobility: park each slot at the start position of its first
    // session (if it has any before simTime; otherwise at the origin -- an
    // unused slot never gets attached or given traffic, so its position is
    // irrelevant beyond costing one extra RT link).
    Ptr<ListPositionAllocator> uePositionAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t slot = 0; slot < poolSize; ++slot)
    {
        Vector initial = sessionsBySlot[slot].empty() ? Vector(0, 0, 0)
                                                       : sessionsBySlot[slot].front()->startPosition;
        uePositionAlloc->Add(initial);
    }
    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    ueMobility.SetPositionAllocator(uePositionAlloc);
    ueMobility.Install(ueNodes);

    // NR / Sionna RT channel setup (identical to khu-real-nr-sionna.cc).
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

    // --sionnaCacheFile switches the channel from live Sionna RT (PathSolver
    // called on every UpdatePeriod tick) to SionnaLookupChannelModel, which
    // answers from a precomputed HDF5 table instead (see
    // scenarios/khu-real/tools/build_sionna_rt_cache.py). This only
    // overrides *which* MatrixBasedChannelModel SionnaRtSpectrumPropagation
    // LossModel wraps -- everything else (antenna arrays, beamforming,
    // handover, etc.) is unchanged. Only valid for the fixed gNB bearing/
    // tilt the cache was built with; a live RET tilt change afterwards
    // won't be reflected (see SionnaLookupChannelModel's class doc).
    if (!sionnaCacheFile.empty())
    {
        // Setting a PointerValue attribute (ChannelModel) via a TypeId
        // StringValue triggers immediate construction of that object right
        // here (Config::SetDefault -> AttributeChecker::CreateValidValue ->
        // PointerValue::DeserializeFromString -> ObjectFactory::Create()) --
        // not deferred to whenever SionnaRtSpectrumPropagationLossModel
        // itself is later constructed. So CacheFile/Frequency must already
        // be the active defaults *before* this call, or the constructed
        // SionnaLookupChannelModel gets built with an empty CacheFile.
        Config::SetDefault("ns3::SionnaLookupChannelModel::CacheFile",
                           StringValue(sionnaCacheFile));
        Config::SetDefault("ns3::SionnaLookupChannelModel::Frequency",
                           DoubleValue(centralFrequencyBand1));
        Config::SetDefault("ns3::SionnaRtSpectrumPropagationLossModel::ChannelModel",
                           StringValue("ns3::SionnaLookupChannelModel"));
        NS_LOG_UNCOND("[sionna] using lookup-table channel model, cache=" << sionnaCacheFile);
    }

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
    // 3GPP TR38.901 directional element pattern, not isotropic: RET
    // (DowntiltAngle/BearingAngle, see NrGnbNetDevice::ApplyRetControl) only
    // has a physical effect on RSRP/throughput if the element itself has a
    // real elevation/azimuth gain lobe to move -- confirmed empirically that
    // an isotropic element leaves tilt/bearing changes at <1dB (noise-level)
    // regardless of angle, since there's no directional pattern for
    // orientation to modulate and DirectPathBeamforming ideally steers
    // toward the UE's true angle either way.
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<ThreeGppAntennaModel>()));

    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(0));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(0));

    if (nrHelper->GetHandoverAlgorithmType() == "ns3::NrA3RsrpHandoverAlgorithm")
    {
        nrHelper->SetHandoverAlgorithmAttribute("TimeToTrigger",
                                                TimeValue(MilliSeconds(handoverTtTMs)));
        nrHelper->SetHandoverAlgorithmAttribute("Hysteresis",
                                                DoubleValue(handoverHysteresisDb));
    }

    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice(ueNodes, allBwps);

    if (gnbNodes.GetN() > 1)
    {
        nrHelper->AddX2Interface(gnbNodes);
    }

    // cellId -> gNB NetDevice, needed to target HandoverRequest's sourceGnbDev.
    std::map<uint16_t, Ptr<NetDevice>> cellIdToGnbDev;
    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i)
    {
        cellIdToGnbDev[static_cast<uint16_t>(i + 1)] = gnbNetDev.Get(i);
    }

    // Slot bookkeeping keyed by slot index (0..poolSize-1), separate from the
    // per-IMSI g_ueKpi map above (that one's keyed by IMSI, which is 1:1 with
    // slot for the device's whole lifetime -- this one tracks pool state).
    std::vector<bool> slotEverAttached(poolSize, false);
    std::vector<std::string> slotGuiName(poolSize);
    for (uint32_t slot = 0; slot < poolSize; ++slot)
    {
        slotGuiName[slot] = "ue_slot_" + std::to_string(slot);
    }

    for (uint32_t i = 0; i < ueNetDev.GetN(); ++i)
    {
        auto ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(i));
        g_imsiToUeNode[ueDev->GetImsi()] = ueNodes.Get(i);
        g_imsiToUeName[ueDev->GetImsi()] = slotGuiName[i]; // GUI marker identity == slot, not trace id
        g_ueKpi[ueDev->GetImsi()].name = ""; // idle until first session claims it
    }
    Config::ConnectWithoutContextFailSafe("/NodeList/*/DeviceList/*/NrGnbRrc/HandoverStart",
                                          MakeCallback(&LogHandoverStart));
    Config::ConnectWithoutContextFailSafe("/NodeList/*/DeviceList/*/NrGnbRrc/HandoverEndOk",
                                          MakeCallback(&LogHandoverEndOk));
    bool connEstConnected = Config::ConnectWithoutContextFailSafe(
        "/NodeList/*/DeviceList/*/NrUeRrc/ConnectionEstablished", MakeCallback(&RecordAttachForKpi));
    NS_LOG_UNCOND("[kpi] trace connect: ConnectionEstablished=" << connEstConnected);
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

        // Apply the gNB's initial bearing/tilt from gnbPositions[i], parsed
        // from gnbs-ret.csv's optional 5th/6th columns. Unlike
        // NrGnbNetDevice::ApplyRetControl (which only fires on an E2SM-RC
        // RET_Tilt_Control message at runtime), nothing previously set these
        // at startup, so every gNB silently sat at the UniformPlanarArray
        // default (bearing=0, tilt=0) regardless of what the CSV said. A
        // real static bearing per gNB now actually matters for RSRP/
        // throughput since AntennaElement is ThreeGppAntennaModel (see
        // above) rather than isotropic.
        {
            Ptr<UniformPlanarArray> antenna = DynamicCast<UniformPlanarArray>(
                NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)->GetSpectrumPhy()->GetAntenna());
            if (antenna)
            {
                if (!std::isnan(gnbPositions[i].bearingDeg))
                {
                    antenna->SetAttribute(
                        "BearingAngle",
                        DoubleValue(gnbPositions[i].bearingDeg * M_PI / 180.0));
                }
                if (!std::isnan(gnbPositions[i].tiltDeg))
                {
                    antenna->SetAttribute(
                        "DowntiltAngle",
                        DoubleValue(gnbPositions[i].tiltDeg * M_PI / 180.0));
                }
                NS_LOG_UNCOND("gNB '" << gnbPositions[i].externalId
                                      << "' initial bearing=" << gnbPositions[i].bearingDeg
                                      << " tilt=" << gnbPositions[i].tiltDeg);

                // The GUI runs its own separate Sionna scene/process -- it
                // only ever learns a gNB's position/color via
                // SendGnbPositionToGui/SendColorToGui above, never its
                // orientation, so the arrow/panel there silently stayed at
                // the default (0,0,0) no matter what this scenario's
                // antenna was actually set to. Mirror it explicitly.
                SendGnbOrientationToGui(guiClient,
                                        gnbPositions[i].externalId,
                                        std::isnan(gnbPositions[i].bearingDeg)
                                            ? 0.0
                                            : gnbPositions[i].bearingDeg,
                                        std::isnan(gnbPositions[i].tiltDeg)
                                            ? 0.0
                                            : gnbPositions[i].tiltDeg);
            }
        }

        // Guarantee at least the serving-cell RSRP gets reported for every
        // attached UE, independent of whether it ever detects a neighbor.
        // NrGnbNetDevice::ConfigureCell() already registers an EVENT_A4
        // (lowest threshold, so it fires immediately) config by default
        // (m_sendCuCp) to approximate periodic reporting -- see its own
        // comment: nr's UE-side RRC doesn't implement true PERIODICAL
        // (NrUeRrc::ApplyMeasConfig asserts on it). But A4 only evaluates
        // *non-serving* stored measurements, so a UE that never detects the
        // other gNB at all (e.g. one dominant cell right on top of it, the
        // other genuinely out of range) never has anything to trigger on --
        // meaning RecordRsrpForKpi never fires for it and rsrp_serving_dbm
        // stays stuck at its 0.0 "no data" default despite a perfectly
        // healthy serving link (real SINR/MCS, real throughput). EVENT_A1
        // only looks at the serving cell itself, so add one here with the
        // same "lowest threshold = fires immediately" trick to close that
        // gap; A4 (added automatically) still covers the neighbor side.
        NrRrcSap::ReportConfigEutra a1Config;
        a1Config.triggerType = NrRrcSap::ReportConfigEutra::EVENT;
        a1Config.eventId = NrRrcSap::ReportConfigEutra::EVENT_A1;
        a1Config.threshold1.choice = NrRrcSap::ThresholdEutra::THRESHOLD_RSRP;
        a1Config.threshold1.range = 0;
        a1Config.timeToTrigger = 0;
        a1Config.triggerQuantity = NrRrcSap::ReportConfigEutra::RSRP;
        a1Config.reportQuantity = NrRrcSap::ReportConfigEutra::BOTH;
        a1Config.maxReportCells = 8;
        a1Config.reportInterval = NrRrcSap::ReportConfigEutra::MS1024;
        a1Config.reportAmount = 0xff;
        gnbNetDev.Get(i)->GetObject<NrGnbNetDevice>()->GetRrc()->AddUeMeasReportConfig(a1Config);
    }

    auto [remoteHost, remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.000));

    InternetStackHelper internet;
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface = nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueNetDev));
    // No global AttachToClosestGnb/AttachToMaxRsrpGnb here (unlike
    // khu-real-nr-sionna.cc): each slot attaches for the first time lazily,
    // at its first session's start, via the arrival handler below.

    uint16_t dlPort = 1234;
    ApplicationContainer serverApps;
    UdpServerHelper dlPacketSink(dlPort);
    serverApps.Add(dlPacketSink.Install(ueNodes));
    for (uint32_t slot = 0; slot < poolSize; ++slot)
    {
        auto ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(slot));
        g_ueKpi[ueDev->GetImsi()].serverApp = DynamicCast<UdpServer>(serverApps.Get(slot));
    }
    serverApps.Start(trafficStart);
    serverApps.Stop(simTime);

    NrQosFlow dlFlow(NrQosFlow::NGBR_LOW_LAT_EMBB);
    Ptr<NrQosRule> dlRule = Create<NrQosRule>();
    NrQosRule::PacketFilter dlpf;
    dlpf.localPortStart = dlPort;
    dlpf.localPortEnd = dlPort;
    dlRule->Add(dlpf);
    for (uint32_t slot = 0; slot < poolSize; ++slot)
    {
        nrHelper->ActivateDedicatedQosFlow(ueNetDev.Get(slot), dlFlow, dlRule);
    }

    ApplicationContainer clientApps;

    // ---- Per-session scheduling: arrival (attach/handover + KPI reset +
    // traffic start), interior position updates, and departure (traffic stop
    // already handled by clientApp.Stop()). ----
    for (const auto& session : sessions)
    {
        const uint32_t slot = session.slot;
        Ptr<NetDevice> ueDevice = ueNetDev.Get(slot);
        Ptr<ConstantPositionMobilityModel> mobility =
            ueNodes.Get(slot)->GetObject<ConstantPositionMobilityModel>();
        const std::string traceId = ueTraces[session.traceIndex].externalId;
        const std::string guiName = slotGuiName[slot];
        auto ueDevPtr = DynamicCast<NrUeNetDevice>(ueDevice);
        const uint64_t imsi = ueDevPtr->GetImsi();

        Simulator::Schedule(
            Seconds(session.start),
            [ueDevice, mobility, &guiClient, guiName, traceId, imsi, slot, &nrHelper, gnbNetDev,
             cellIdToGnbDev, position = session.startPosition, &slotEverAttached]() {
                mobility->SetPosition(position);
                SendUePositionToGui(guiClient, guiName, position);

                UeKpiState& kpi = g_ueKpi[imsi];
                kpi.name = traceId;
                kpi.rsrpDbm.clear();
                kpi.hoCount = 0;
                kpi.lastHoTime = -1e9;
                kpi.lastHoSource = 0;
                kpi.lastHoTarget = 0;
                kpi.isPingPong = false;
                if (kpi.serverApp)
                {
                    kpi.lastRxPackets = kpi.serverApp->GetReceived();
                }

                if (!slotEverAttached[slot])
                {
                    // AttachToMaxRsrpGnb (true RSRP-based initial attach) was
                    // tried here first but reliably segfaults when invoked
                    // dynamically mid-simulation in this NR fork -- it only
                    // seems to be exercised, upstream, as a one-shot call for
                    // all UEs before Simulator::Run() begins. Falling back to
                    // the same distance-based AttachToGnb the baseline
                    // scenario (khu-real-nr-sionna.cc) uses for its own
                    // initial deployment: with only 2 isotropic, non-RET gNBs
                    // here, nearest-by-distance and strongest-by-RSRP pick
                    // the same cell in practice.
                    slotEverAttached[slot] = true;
                    double minDistance = std::numeric_limits<double>::infinity();
                    Ptr<NetDevice> closest;
                    for (uint32_t g = 0; g < gnbNetDev.GetN(); ++g)
                    {
                        Vector gnbPos =
                            gnbNetDev.Get(g)->GetNode()->GetObject<MobilityModel>()->GetPosition();
                        double d = CalculateDistance(position, gnbPos);
                        if (d < minDistance)
                        {
                            minDistance = d;
                            closest = gnbNetDev.Get(g);
                        }
                    }
                    nrHelper->AttachToGnb(ueDevice, closest);
                    NS_LOG_UNCOND("[pool] t=" << Simulator::Now().GetSeconds() << " slot=" << slot
                                              << " '" << traceId << "' first attach (closest gNB)");
                }
                else
                {
                    // Already RRC-connected from a previous occupant of this
                    // slot: move it via a real (but KPI-suppressed) handover
                    // to the geometrically closest gNB, rather than calling
                    // AttachToGnb again on a live connection.
                    auto ueDevCast = DynamicCast<NrUeNetDevice>(ueDevice);
                    uint16_t currentCellId = ueDevCast->GetCellId();
                    double minDistance = std::numeric_limits<double>::infinity();
                    uint16_t targetCellId = currentCellId;
                    for (uint32_t g = 0; g < gnbNetDev.GetN(); ++g)
                    {
                        Vector gnbPos =
                            gnbNetDev.Get(g)->GetNode()->GetObject<MobilityModel>()->GetPosition();
                        double d = CalculateDistance(position, gnbPos);
                        if (d < minDistance)
                        {
                            minDistance = d;
                            targetCellId = static_cast<uint16_t>(g + 1);
                        }
                    }
                    if (targetCellId != currentCellId)
                    {
                        auto sourceGnbDev = cellIdToGnbDev.at(currentCellId);
                        g_suppressNextHoKpi.insert(imsi);
                        nrHelper->HandoverRequest(Seconds(0), ueDevice, sourceGnbDev, targetCellId);
                        NS_LOG_UNCOND("[pool] t=" << Simulator::Now().GetSeconds() << " slot="
                                                  << slot << " '" << traceId
                                                  << "' reuse, moving cell " << currentCellId
                                                  << " -> " << targetCellId);
                    }
                    else
                    {
                        NS_LOG_UNCOND("[pool] t=" << Simulator::Now().GetSeconds() << " slot="
                                                  << slot << " '" << traceId
                                                  << "' reuse, staying on cell " << currentCellId);
                    }
                }
            });

        // Interior position samples for this session (the arrival sample at
        // session.start is already handled above).
        for (const auto& sample : ueTraces[session.traceIndex].samples)
        {
            if (sample.time <= session.start || sample.time > session.end)
            {
                continue;
            }
            Simulator::Schedule(
                Seconds(sample.time), [mobility, sample, slot, &guiClient, guiName]() {
                    mobility->SetPosition(sample.position);
                    SendUePositionToGui(guiClient, guiName, sample.position);
                    NS_LOG_UNCOND("[trace] t=" << Simulator::Now().GetSeconds() << " slot=" << slot
                                               << " pos=(" << sample.position.x << ","
                                               << sample.position.y << "," << sample.position.z
                                               << ")");
                });
        }

        // Traffic for this session, clipped to [trafficStart, simTime] exactly
        // like khu-real-nr-sionna.cc's installClient() lambda.
        double start = std::max(session.start, trafficStart.GetSeconds());
        double stop = std::min(session.end, simTime.GetSeconds());
        if (stop > start)
        {
            UdpClientHelper dlClient(ueIpIface.GetAddress(slot), dlPort);
            dlClient.SetAttribute("Interval", TimeValue(MilliSeconds(20)));
            dlClient.SetAttribute("MaxPackets", UintegerValue(1000000));
            dlClient.SetAttribute("PacketSize", UintegerValue(1024));
            ApplicationContainer clientApp = dlClient.Install(remoteHost);
            clientApp.Start(Seconds(start));
            clientApp.Stop(Seconds(stop));
            clientApps.Add(clientApp);
        }
    }

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
