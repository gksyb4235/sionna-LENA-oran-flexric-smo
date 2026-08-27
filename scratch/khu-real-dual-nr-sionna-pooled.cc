// Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

/**
 * @file khu-real-dual-nr-sionna-pooled.cc
 * @brief Dual-band NR variant of khu-real-nr-sionna-pooled.cc. gNBs whose CSV
 * label starts with gNB_4G use a 1.8 GHz NR carrier and gNB_5G uses a 3.5 GHz
 * NR carrier. Both carriers use numerology 0; the labels are retained for GUI
 * and KPI compatibility and do not denote an LTE radio stack.
 *
 * Active windows are precomputed before the run. By default logical sessions
 * reuse a bounded pool of UE/RRC objects with a short post-departure guard.
 * A disappearing logical UE is removed from KPI reporting immediately. Late
 * handover callbacks carry a session generation and therefore cannot be
 * attributed to the next logical subscriber using the same physical slot.
 *
 * This is a separate scratch target from khu-real-nr-sionna.cc -- nothing in
 * ns-3 core, contrib/nr, or the original scenario file is touched.
 *
 * Slot assignment is precomputed once, offline, before Simulator::Run(), via
 * classic interval-graph-coloring (greedy, earliest-available-slot): sort all
 * (trace, active-window) sessions by start time, reuse the slot whose
 * previous session ended earliest plus the configured guard if it's already
 * free by the new session's start, otherwise allocate a new slot. This keeps
 * the schedule static, so no runtime free-list is needed in Simulator::Run().
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
 * Only a slot's first session performs a closest-gNB attach. Later sessions
 * retain the physical RRC connection, move the slot to their first position,
 * and let normal A3 measurements trigger any required handover. Session
 * departure clears its logical KPI identity. A short settling window excludes
 * the slot-relocation HO from KPI accounting; only later mobility handovers
 * that start and finish in the same session contribute to HO/ping-pong counts.
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
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <execinfo.h>
#include <functional>
#include <fstream>
#include <iterator>
#include <limits>
#include <queue>
#include <sstream>
#include <utility>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("KhuRealDualNrSionnaPooled");

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

enum class NrBandKind
{
    LOW_18_GHZ,
    MID_35_GHZ,
};

static NrBandKind
GetBandKind(const std::string& externalId)
{
    if (externalId.rfind("gNB_4G", 0) == 0)
    {
        return NrBandKind::LOW_18_GHZ;
    }
    if (externalId.rfind("gNB_5G", 0) == 0)
    {
        return NrBandKind::MID_35_GHZ;
    }
    NS_ABORT_MSG("gNB '" << externalId
                          << "' must start with gNB_4G or gNB_5G to select its NR band");
    return NrBandKind::MID_35_GHZ;
}

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
ValidateSionnaCacheTopology(const std::string& cacheFile,
                            const std::vector<GnbPosition>& liveGnbs,
                            NrBandKind bandKind,
                            double expectedFrequency)
{
    NS_ABORT_MSG_IF(cacheFile.empty(), "Sionna cache path is empty");
    try
    {
        py::object h5 = py::module_::import("h5py").attr("File")(cacheFile, "r");
        double cachedFrequency = h5.attr("attrs")["frequency_hz"].cast<double>();
        NS_ABORT_MSG_IF(std::abs(cachedFrequency - expectedFrequency) > 1.0,
                        "Sionna cache frequency mismatch: file=" << cacheFile
                                                                  << " cache=" << cachedFrequency
                                                                  << " live=" << expectedFrequency);

        for (const auto& gnb : liveGnbs)
        {
            if (GetBandKind(gnb.externalId) != bandKind)
            {
                continue;
            }
            bool hasGroup = h5.attr("__contains__")(gnb.externalId).cast<bool>();
            NS_ABORT_MSG_IF(!hasGroup,
                            "Sionna cache " << cacheFile << " has no group " << gnb.externalId);
            py::object cachedPosition =
                h5.attr("__getitem__")(gnb.externalId).attr("attrs")["gnb_position"];
            Vector cachePos(cachedPosition.attr("__getitem__")(0).cast<double>(),
                            cachedPosition.attr("__getitem__")(1).cast<double>(),
                            cachedPosition.attr("__getitem__")(2).cast<double>());
            double offset = CalculateDistance(cachePos, gnb.position);
            NS_ABORT_MSG_IF(offset >= 0.05,
                            "Sionna cache/live gNB position mismatch for "
                                << gnb.externalId << ": cache=(" << cachePos.x << ',' << cachePos.y
                                << ',' << cachePos.z << ") live=(" << gnb.position.x << ','
                                << gnb.position.y << ',' << gnb.position.z << ") offset=" << offset
                                << "m. Rebuild " << cacheFile << " from the current gnbs-ret.csv");
        }
        h5.attr("close")();
    }
    catch (const py::error_already_set& error)
    {
        NS_ABORT_MSG("Failed to validate Sionna cache " << cacheFile << ": " << error.what());
    }
}

static void
ConfigureSionnaLookupCache(const BandwidthPartInfoPtrVector& bwps,
                           const std::string& cacheFile,
                           double frequency,
                           Time updatePeriod,
                           const std::string& bandLabel)
{
    NS_ABORT_MSG_IF(cacheFile.empty(), "Missing Sionna cache for " << bandLabel);

    for (const auto& bwp : bwps)
    {
        Ptr<SpectrumChannel> spectrumChannel = bwp.get()->GetChannel();
        NS_ABORT_MSG_IF(!spectrumChannel, "No spectrum channel for " << bandLabel);
        Ptr<PhasedArraySpectrumPropagationLossModel> phasedArrayChannel =
            spectrumChannel->GetPhasedArraySpectrumPropagationLossModel();
        NS_ABORT_MSG_IF(!phasedArrayChannel,
                        "No phased-array spectrum propagation model for " << bandLabel);
        Ptr<SionnaRtSpectrumPropagationLossModel> sionna =
            phasedArrayChannel->GetObject<SionnaRtSpectrumPropagationLossModel>();
        NS_ABORT_MSG_IF(!sionna, "No Sionna spectrum propagation model for " << bandLabel);

        Ptr<SionnaLookupChannelModel> lookup = CreateObject<SionnaLookupChannelModel>();
        lookup->SetCacheFile(cacheFile);
        lookup->SetFrequency(frequency);
        lookup->SetUpdatePeriod(updatePeriod);
        sionna->SetChannelModel(lookup);
    }

    NS_LOG_UNCOND("[sionna] " << bandLabel << " frequency=" << frequency
                               << "Hz cache=" << cacheFile
                               << " updatePeriod=" << updatePeriod.As(Time::S));
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

// Keyed by gNB external id (not CSV row order) so a given cell name always
// gets the same color in both the GUI and the Grafana "Connected UEs" panel
// (monitoring/grafana/dashboards/cell_kpi.json field overrides use these same
// hex values). Falls back to a rotating palette for any unlisted name.
static const std::map<std::string, Vector> kCellColorByName = {
    {"gNB_5G", Vector(0.90, 0.20, 0.20)},   // red   #E63333
    {"gNB_4G_1", Vector(0.20, 0.45, 0.90)}, // blue  #3373E6
    {"gNB_4G_2", Vector(0.20, 0.75, 0.35)}, // green #33BF59
};

static const std::vector<Vector> kCellColorFallbackPalette = {
    Vector(0.95, 0.60, 0.10), // orange
    Vector(0.65, 0.30, 0.85), // purple
    Vector(0.20, 0.80, 0.80), // teal
};

static Vector
CellColorForName(const std::string& externalId, uint32_t fallbackIndex)
{
    auto it = kCellColorByName.find(externalId);
    if (it != kCellColorByName.end())
    {
        return it->second;
    }
    return kCellColorFallbackPalette[fallbackIndex % kCellColorFallbackPalette.size()];
}

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
    uint16_t completedServingCellId = 0;
    double dlSinrDb = std::numeric_limits<double>::quiet_NaN();
    int16_t dlMcs = -1;
    uint64_t sessionGeneration = 0; //!< prevents late HO callbacks crossing logical sessions
    double suppressHoAccountingUntil = -1e9; //!< excludes slot-relocation settling HO
};

static std::map<uint64_t, UeKpiState> g_ueKpi; //!< keyed by IMSI (== keyed by slot, 1 IMSI/slot for life)

// One row per (report tick, UE) / (report tick, cell), same fields and same
// per-tick cadence (kpiReportInterval) as what ReportKpiToInflux sends to
// InfluxDB -- buffered here so the exact same data is available as CSV at
// the end of the run regardless of whether InfluxDB is reachable/enabled.
static std::ofstream g_ueKpiCsv;
static std::ofstream g_cellKpiCsv;
static uint64_t g_ueKpiRowCount = 0;
static uint64_t g_cellKpiRowCount = 0;

// Opens both KPI CSVs and writes their headers immediately; every row is
// then written and flushed as it's produced (see the two write sites in
// ReportKpiToInflux below), instead of buffering the whole run in memory
// and dumping it once after Simulator::Run() returns. An uncaught exception
// mid-run (std::terminate) skips everything after Simulator::Run(), which
// used to mean a crash lost the entire run's KPI history even if 99% of it
// had already completed.
static void
OpenKpiCsvFiles(const std::string& ueCsvPath, const std::string& cellCsvPath)
{
    g_ueKpiCsv.open(ueCsvPath);
    g_ueKpiCsv << "time_s,ue,serving_cell,rsrp_serving_dbm,neighbor_cell,rsrp_neighbor_dbm,"
                  "dl_throughput_mbps,ho_count,seconds_since_ho,is_pingpong,dl_sinr_db,dl_mcs\n";
    g_ueKpiCsv.flush();

    g_cellKpiCsv.open(cellCsvPath);
    g_cellKpiCsv << "time_s,cell,tx_power_dbm,ret_tilt_deg,ret_bearing_deg,ttt_ms,hysteresis_db,"
                    "num_ues,avg_rsrp_dbm,ho_in_count,ho_out_count,pingpong_count,"
                    "avg_throughput_mbps,prb_utilization_pct,energy_state,tx_power_watts,"
                    "energy_efficiency_mbps_per_w\n";
    g_cellKpiCsv.flush();
}

// Installed via std::set_terminate so an uncaught exception (e.g. the
// std::out_of_range from a stray vector::at()) leaves behind the exception
// message and a raw backtrace instead of just glibc's bare "terminate
// called after throwing an instance of ...". Frame addresses are printed as
// module+offset (via dladdr) since gdb isn't available in this environment;
// resolve file:line offline with `addr2line -e <module> -f -C <offset>` --
// the binary is built with debug_info. Also flushes the KPI CSVs one last
// time so whatever was already written survives (they're flushed after
// every row anyway, see OpenKpiCsvFiles, so this is a belt-and-suspenders
// last resort).
static void
CrashTerminateHandler()
{
    std::cerr << "\n[FATAL] std::terminate called";
    if (auto ex = std::current_exception())
    {
        try
        {
            std::rethrow_exception(ex);
        }
        catch (const std::exception& e)
        {
            std::cerr << " -- uncaught exception: " << e.what();
        }
        catch (...)
        {
            std::cerr << " -- uncaught exception of unknown type";
        }
    }
    std::cerr << " at simTime=" << Simulator::Now().GetSeconds() << "s\n";

    void* frames[64];
    int n = backtrace(frames, 64);
    std::cerr << "[FATAL] backtrace (" << n << " frames):\n";
    for (int i = 0; i < n; ++i)
    {
        Dl_info info;
        if (dladdr(frames[i], &info) && info.dli_fname)
        {
            uintptr_t offset = reinterpret_cast<uintptr_t>(frames[i]) -
                                reinterpret_cast<uintptr_t>(info.dli_fbase);
            std::cerr << "  #" << i << " " << info.dli_fname << "+0x" << std::hex << offset
                      << std::dec;
            if (info.dli_sname)
            {
                std::cerr << " (" << info.dli_sname << ")";
            }
            std::cerr << "\n";
        }
        else
        {
            std::cerr << "  #" << i << " " << frames[i] << " (unresolved)\n";
        }
    }

    g_ueKpiCsv.flush();
    g_cellKpiCsv.flush();

    std::abort();
}

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

// Cumulative-since-start-of-sim cell counters (hoInCount/hoOutCount/
// pingPongCount are never reset elsewhere, see LogHandoverEndOk), printed
// once at the very end instead of live per-second InfluxDB writes -- lets a
// run without --influxSrc (or with InfluxDB down) still leave a summary
// instead of nothing at all.
static void
PrintFinalKpiSummary()
{
    std::cout << "\n=== Final Cell KPI Summary ===\n";
    uint32_t totalHo = 0;
    uint32_t totalPingPong = 0;
    for (const auto& [cellId, cell] : g_cellKpi)
    {
        std::cout << "  " << cell.name << ": ho_in=" << cell.hoInCount
                  << " ho_out=" << cell.hoOutCount << " ping_pong=" << cell.pingPongCount << "\n";
        totalHo += cell.hoInCount;
        totalPingPong += cell.pingPongCount;
    }
    std::cout << "  TOTAL: handovers=" << totalHo << " ping_pongs=" << totalPingPong << "\n";
}

static std::map<std::pair<uint16_t, uint16_t>, uint64_t> g_cellRntiToImsi;
static std::map<uint16_t, uint64_t> g_rntiToImsi;

static std::map<uint64_t, uint16_t> g_pendingHoSourceCell;
static std::map<uint64_t, uint64_t> g_pendingHoSessionGeneration;
static std::map<uint64_t, bool> g_pendingHoAccountingSuppressed;
static double g_handoverTtTMs = 0.0;
static double g_handoverHysteresisDb = 0.0;
static double g_pingPongWindowSec = 3.0;

static void
ClearRadioKpiMappingsForImsi(uint64_t imsi)
{
    for (auto it = g_cellRntiToImsi.begin(); it != g_cellRntiToImsi.end();)
    {
        it = (it->second == imsi) ? g_cellRntiToImsi.erase(it) : std::next(it);
    }
    for (auto it = g_rntiToImsi.begin(); it != g_rntiToImsi.end();)
    {
        it = (it->second == imsi) ? g_rntiToImsi.erase(it) : std::next(it);
    }
}

static bool
CanDisconnectPooledUe(NrUeRrc::State state)
{
    return state != NrUeRrc::IDLE_WAIT_SIB2 && state != NrUeRrc::IDLE_RANDOM_ACCESS &&
           state != NrUeRrc::IDLE_CONNECTING;
}

static void
AttachSessionUe(Ptr<NrHelper> nrHelper,
                Ptr<NetDevice> ueDevice,
                Ptr<NetDevice> targetGnb,
                uint32_t slot,
                const std::string& traceId)
{
    Ptr<NrUeNetDevice> ue = DynamicCast<NrUeNetDevice>(ueDevice);
    NS_ABORT_MSG_IF(!ue, "session slot " << slot << " is not an NrUeNetDevice");

    nrHelper->AttachToGnb(ueDevice, targetGnb);
    NS_LOG_UNCOND("[pool] t=" << Simulator::Now().GetSeconds() << " slot=" << slot << " '"
                              << traceId << "' first attach to cell "
                              << DynamicCast<NrGnbNetDevice>(targetGnb)->GetCellId());
}

static void
DisconnectIdlePooledUe(Ptr<NetDevice> ueDevice,
                       uint64_t imsi,
                       uint32_t slot,
                       uint32_t retryCount = 0)
{
    auto kpiIt = g_ueKpi.find(imsi);
    if (kpiIt == g_ueKpi.end() || !kpiIt->second.name.empty())
    {
        return;
    }

    Ptr<NrUeNetDevice> ue = DynamicCast<NrUeNetDevice>(ueDevice);
    NrUeRrc::State state = ue->GetRrc()->GetState();
    if (!CanDisconnectPooledUe(state))
    {
        constexpr uint32_t kMaxRetries = 1000;
        NS_ABORT_MSG_IF(retryCount >= kMaxRetries,
                        "idle pool slot " << slot << " remained in transient RRC state "
                                          << static_cast<int>(state) << " for 10 seconds");
        Simulator::Schedule(MilliSeconds(10),
                            &DisconnectIdlePooledUe,
                            ueDevice,
                            imsi,
                            slot,
                            retryCount + 1);
        return;
    }

    ue->GetNas()->Disconnect();
    g_pendingHoSourceCell.erase(imsi);
    g_pendingHoSessionGeneration.erase(imsi);
    g_pendingHoAccountingSuppressed.erase(imsi);
    ClearRadioKpiMappingsForImsi(imsi);
    NS_LOG_UNCOND("[pool] t=" << Simulator::Now().GetSeconds() << " slot=" << slot
                              << " disconnected after departure");
}

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
    if (auto it = g_ueKpi.find(imsi); it != g_ueKpi.end())
    {
        it->second.completedServingCellId = cellId;
    }
    // ConnectionEstablished fires for a UE's very first attach (AttachToGnb),
    // which never goes through HandoverStart/HandoverEndOk -- without this,
    // a freshly-attached UE stays at the GUI's default color until/unless it
    // later happens to handover, even though it's already really being
    // served by cellId.
    RecolorUeForCell(imsi, cellId);
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
    ueIt->second.dlMcs = static_cast<int16_t>(mcs);
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
    const bool influxAvailable = g_influxClient && !g_influxClient->is_none();
    const double now = Simulator::Now().GetSeconds();

    std::map<uint16_t, uint32_t> cellUeCount;
    std::map<uint16_t, uint32_t> cellRsrpCount;
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
        // NrUeNetDevice::GetCellId() follows the target gNB pointer, which can
        // change before a handover has completed. Report only the cell confirmed
        // by ConnectionEstablished/HandoverEndOk so an in-flight or failed HO is
        // not presented as a successful serving-cell transition.
        uint16_t servingCellId = ue.completedServingCellId;

        double throughputMbps = 0.0;
        if (ue.serverApp)
        {
            uint64_t rx = ue.serverApp->GetReceived();
            uint64_t deltaPackets = (rx >= ue.lastRxPackets) ? (rx - ue.lastRxPackets) : 0;
            ue.lastRxPackets = rx;
            throughputMbps = (deltaPackets * 1024.0 * 8.0) / (intervalSec * 1e6);
        }

        double rsrpServing = std::numeric_limits<double>::quiet_NaN();
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

        if (influxAvailable)
        {
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
        }

        g_ueKpiCsv << now << ',' << ue.name << ',' << servingCellName << ',' << rsrpServing << ','
                   << bestNeighborName << ',' << bestNeighborRsrp << ',' << throughputMbps << ','
                   << ue.hoCount << ',' << secondsSinceHo << ',' << (ue.isPingPong ? 1 : 0) << ','
                   << ue.dlSinrDb << ',' << static_cast<int>(ue.dlMcs) << '\n';
        g_ueKpiCsv.flush();
        ++g_ueKpiRowCount;

        if (g_cellKpi.count(servingCellId))
        {
            cellUeCount[servingCellId]++;
            if (std::isfinite(rsrpServing))
            {
                cellRsrpSum[servingCellId] += rsrpServing;
                cellRsrpCount[servingCellId]++;
            }
            cellThroughputSum[servingCellId] += throughputMbps;
        }
    }

    for (auto& cellEntry : g_cellKpi)
    {
        uint16_t cellId = cellEntry.first;
        CellKpiState& cell = cellEntry.second;
        uint32_t numUes = cellUeCount.count(cellId) ? cellUeCount[cellId] : 0;
        uint32_t numRsrpSamples = cellRsrpCount.count(cellId) ? cellRsrpCount[cellId] : 0;
        double avgRsrp = (numRsrpSamples > 0)
                             ? (cellRsrpSum[cellId] / numRsrpSamples)
                             : std::numeric_limits<double>::quiet_NaN();
        double cellThroughputTotal =
            cellThroughputSum.count(cellId) ? cellThroughputSum[cellId] : 0.0;
        double avgThroughput = (numUes > 0) ? (cellThroughputTotal / numUes) : 0.0;

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

        // Linear (EARTH-style) power model: P = P0 + deltaP * Ptx when the
        // cell is ON, so RET/scheduling-driven throughput changes never move
        // power on their own -- only TxPower does, via ApplyEnergyState's
        // -100dBm (OFF) / -20dB (SLEEP) knobs. OFF/SLEEP use fixed low
        // wattages instead of the linear formula: at -100dBm, Ptx is ~0W, so
        // the formula would otherwise still report near-full P0 baseband
        // overhead as if the radio chain were still fully powered up.
        constexpr double kP0Watts = 130.0;
        constexpr double kDeltaP = 4.7;
        constexpr double kSleepPowerWatts = 30.0;
        constexpr double kOffPowerWatts = 5.0;
        std::string energyState = "ON";
        if (cell.gnbDev)
        {
            energyState = DynamicCast<NrGnbNetDevice>(cell.gnbDev)->GetEnergyStateName();
        }
        double totalPowerWatts;
        if (energyState == "OFF")
        {
            totalPowerWatts = kOffPowerWatts;
        }
        else if (energyState == "SLEEP")
        {
            totalPowerWatts = kSleepPowerWatts;
        }
        else
        {
            double txPowerWatts = pow(10.0, liveTxPowerDbm / 10.0) / 1000.0;
            totalPowerWatts = kP0Watts + kDeltaP * txPowerWatts;
        }
        double energyEfficiencyMbpsPerW =
            (totalPowerWatts > 0.0) ? (cellThroughputTotal / totalPowerWatts) : 0.0;

        if (influxAvailable)
        {
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
                                                       avgThroughput,
                                                       prbUtilizationPct,
                                                       energyState,
                                                       totalPowerWatts,
                                                       energyEfficiencyMbpsPerW);
            }
            catch (const py::error_already_set& error)
            {
                NS_LOG_UNCOND("[influx] write_cell_kpi failed (ignoring): " << error.what());
            }
        }

        g_cellKpiCsv << now << ',' << cell.name << ',' << liveTxPowerDbm << ',' << retTiltDeg
                     << ',' << retBearingDeg << ',' << g_handoverTtTMs << ','
                     << g_handoverHysteresisDb << ',' << static_cast<int>(numUes) << ','
                     << avgRsrp << ',' << cell.hoInCount << ',' << cell.hoOutCount << ','
                     << cell.pingPongCount << ',' << avgThroughput << ',' << prbUtilizationPct
                     << ',' << energyState << ',' << totalPowerWatts << ','
                     << energyEfficiencyMbpsPerW << '\n';
        g_cellKpiCsv.flush();
        ++g_cellKpiRowCount;
    }

    Simulator::Schedule(Seconds(intervalSec), &ReportKpiToInflux, ueNetDev, intervalSec);
}

static void
LogHandoverStart(uint64_t imsi, uint16_t sourceCellId, uint16_t rnti, uint16_t targetCellId)
{
    g_pendingHoSourceCell[imsi] = sourceCellId;
    g_pendingHoSessionGeneration.erase(imsi);
    g_pendingHoAccountingSuppressed.erase(imsi);
    if (auto ueIt = g_ueKpi.find(imsi);
        ueIt != g_ueKpi.end() && !ueIt->second.name.empty())
    {
        g_pendingHoSessionGeneration[imsi] = ueIt->second.sessionGeneration;
        g_pendingHoAccountingSuppressed[imsi] =
            Simulator::Now().GetSeconds() < ueIt->second.suppressHoAccountingUntil;
    }
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
    uint64_t pendingSessionGeneration = 0;
    bool accountingSuppressed = false;
    if (auto it = g_pendingHoSourceCell.find(imsi); it != g_pendingHoSourceCell.end())
    {
        sourceCellId = it->second;
        g_pendingHoSourceCell.erase(it);
    }
    if (auto it = g_pendingHoSessionGeneration.find(imsi);
        it != g_pendingHoSessionGeneration.end())
    {
        pendingSessionGeneration = it->second;
        g_pendingHoSessionGeneration.erase(it);
    }
    if (auto it = g_pendingHoAccountingSuppressed.find(imsi);
        it != g_pendingHoAccountingSuppressed.end())
    {
        accountingSuppressed = it->second;
        g_pendingHoAccountingSuppressed.erase(it);
    }

    if (auto ueIt = g_ueKpi.find(imsi); ueIt != g_ueKpi.end())
    {
        UeKpiState& ue = ueIt->second;
        const bool belongsToCurrentSession = !ue.name.empty() &&
                                             pendingSessionGeneration != 0 &&
                                             pendingSessionGeneration == ue.sessionGeneration;
        if (belongsToCurrentSession && !accountingSuppressed)
        {
            double now = Simulator::Now().GetSeconds();
            ue.isPingPong =
                (ue.hoCount > 0) && (sourceCellId == ue.lastHoTarget) &&
                (cellId == ue.lastHoSource) &&
                (now - ue.lastHoTime) < g_pingPongWindowSec;
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
    }

    Vector pos(0, 0, 0);
    if (auto it = g_imsiToUeNode.find(imsi); it != g_imsiToUeNode.end())
    {
        pos = it->second->GetObject<MobilityModel>()->GetPosition();
    }
    NS_LOG_UNCOND("[HO] t=" << Simulator::Now().GetSeconds() << "s END_OK imsi=" << imsi
                            << " rnti=" << rnti << " now_serving_cell=" << cellId
                            << (accountingSuppressed ? " kpi=suppressed-slot-settling" : "")
                            << " ue_pos=(" << pos.x << "," << pos.y << "," << pos.z << ")");
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
        // No explicit active=false departure row (current ue_positions_seed*
        // .csv traces don't emit one -- a UE's window just stops appearing).
        // Close the session at that UE's own last sample time, not blindly
        // at simTimeSec -- otherwise every UE that never gets an explicit
        // departure row (all of them, in traces with no "false" rows at all)
        // looks "active" all the way to the end of the simulation, collapsing
        // the whole point of slot pooling (observed: max concurrent silently
        // inflating from 85 to 262 on a trace with zero active=false rows).
        double lastSampleTime = trace.samples.empty() ? activeStart : trace.samples.back().time;
        windows.emplace_back(activeStart, std::min(lastSampleTime, simTimeSec));
        startPositions->push_back(activeStartPos);
    }
    return windows;
}

/// Builds all sessions and either gives each one an isolated device or assigns
/// minimum-size reusable slots via greedy interval-graph coloring.
static std::vector<UeSession>
BuildSessionsAndAssignSlots(const std::vector<UeTrace>& traces,
                            double simTimeSec,
                            bool reuseSlots,
                            double slotReuseGuardSec,
                            uint32_t* poolSize,
                            uint32_t* maxConcurrent)
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

    // Min-heap of (available time, slot id). First calculate the exact
    // simultaneous-active count without a guard, then assign the requested
    // reusable pool with a small post-departure guard. The guard prevents a
    // late RRC/HO callback from the old logical subscriber from landing after
    // the slot has already been claimed by a new one.
    using FreeAtEntry = std::pair<double, uint32_t>;
    auto assignGreedy = [&sessions](double guardSec, bool writeSlots) {
        std::priority_queue<FreeAtEntry, std::vector<FreeAtEntry>, std::greater<>> freeAt;
        uint32_t nextSlot = 0;
        for (auto& session : sessions)
        {
            uint32_t slot;
            if (!freeAt.empty() && freeAt.top().first <= session.start)
            {
                slot = freeAt.top().second;
                freeAt.pop();
            }
            else
            {
                slot = nextSlot++;
            }
            if (writeSlots)
            {
                session.slot = slot;
            }
            freeAt.push({session.end + guardSec, slot});
        }
        return nextSlot;
    };

    *maxConcurrent = assignGreedy(0.0, false);
    if (reuseSlots)
    {
        *poolSize = assignGreedy(slotReuseGuardSec, true);
    }
    else
    {
        for (uint32_t i = 0; i < sessions.size(); ++i)
        {
            sessions[i].slot = i;
        }
        *poolSize = sessions.size();
    }
    return sessions;
}

int
main(int argc, char* argv[])
{
    std::set_terminate(&CrashTerminateHandler);

    py::scoped_interpreter guard{};

    std::string sionnaCacheFile18;
    std::string sionnaCacheFile35;
    std::string gnbPositionsPath = "scenarios/khu-real/gnbs-ret.csv";
    std::string sumoTracePath = "scenarios/khu-real/ue_positions_seed0.csv";
    uint32_t nUes = 300; // upper bound on distinct real ue_id count in the trace CSV

    Time simTime = Seconds(900);
    Time trafficStart = Seconds(0.5);
    bool requireTraffic = false;
    Time sionnaUpdatePeriod = Seconds(1);

    double centralFrequency18 = 1.8e9;
    double bandwidth18 = 10e6;
    double centralFrequency35 = 3.5e9;
    double bandwidth35 = 15e6;
    uint16_t numerology = 0;
    double totalTxPower = 43;

    std::string guiSrc;
    std::string guiHost = "localhost";

    std::string influxSrc;
    std::string influxHost = "localhost";
    uint16_t influxPort = 8086;
    std::string influxDb = "nr_kpi";
    double kpiReportInterval = 1.0;

    double handoverTtTMs = 256.0;
    double handoverHysteresisDb = 3.0;
    double pingPongWindowSec = 3.0;

    bool trafficMultiplierEnabled = true;
    double udpBaseIntervalMs = 200.0;
    bool reuseUeSlots = true;
    double slotReuseGuardSec = 1.0;
    double slotSettlingHoWindowSec = 2.0;

    std::string ueKpiCsvPath = "ue_kpi.csv";
    std::string cellKpiCsvPath = "cell_kpi.csv";

    uint32_t poolSizeOverride = 0; // 0 = auto for the selected isolation/reuse mode

    CommandLine cmd(__FILE__);
    cmd.AddValue("sionnaCacheFile18",
                 "1.8 GHz HDF5 cache built for gNB 2x2 and UE 1x1 arrays",
                 sionnaCacheFile18);
    cmd.AddValue("sionnaCacheFile35",
                 "3.5 GHz HDF5 cache built for gNB 2x2 and UE 1x1 arrays",
                 sionnaCacheFile35);
    cmd.AddValue("gnbPositions", "gNB position CSV path", gnbPositionsPath);
    cmd.AddValue("sumoTrace", "UE trace CSV path", sumoTracePath);
    cmd.AddValue("N_Ues", "Upper bound on distinct ue_id count in the trace", nUes);
    cmd.AddValue("poolSize",
                 "Fixed UE device count (0 = auto for the selected isolation/reuse mode)",
                 poolSizeOverride);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.AddValue("trafficStart", "Downlink UDP traffic start time", trafficStart);
    cmd.AddValue("requireTraffic",
                 "Return non-zero status if no downlink bytes reach any UE",
                 requireTraffic);
    cmd.AddValue("sionnaUpdatePeriod", "Sionna lookup-channel update period", sionnaUpdatePeriod);
    cmd.AddValue("centralFrequency18", "Low-band NR center frequency in Hz", centralFrequency18);
    cmd.AddValue("bandwidth18", "Low-band NR bandwidth in Hz", bandwidth18);
    cmd.AddValue("centralFrequency35", "Mid-band NR center frequency in Hz", centralFrequency35);
    cmd.AddValue("bandwidth35", "Mid-band NR bandwidth in Hz", bandwidth35);
    cmd.AddValue("numerology", "Common numerology for both NR bands", numerology);
    cmd.AddValue("totalTxPower", "Total gNB tx power (dBm)", totalTxPower);
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
    cmd.AddValue("pingPongWindow",
                 "Reverse-HO interval in seconds counted as ping-pong (statistics only)",
                 pingPongWindowSec);
    cmd.AddValue("reuseUeSlots",
                 "Reuse a bounded pool of connected UE devices across logical trace sessions",
                 reuseUeSlots);
    cmd.AddValue("slotReuseGuard",
                 "Seconds between one logical UE departing and reuse of its physical slot",
                 slotReuseGuardSec);
    cmd.AddValue("slotSettlingHoWindow",
                 "Seconds after slot reuse during which relocation HO is excluded from KPI",
                 slotSettlingHoWindowSec);
    cmd.AddValue("trafficMultiplierEnabled",
                "Randomize per-session traffic 2x-20x (false = flat base rate)",
                trafficMultiplierEnabled);
    cmd.AddValue("udpBaseIntervalMs",
                 "Base UDP packet interval in ms before applying the 2x-20x multiplier",
                 udpBaseIntervalMs);
    cmd.AddValue("ueKpiCsvPath", "Output CSV path for per-second UE KPI history", ueKpiCsvPath);
    cmd.AddValue("cellKpiCsvPath",
                "Output CSV path for per-second cell KPI history",
                cellKpiCsvPath);
    cmd.Parse(argc, argv);

    g_handoverTtTMs = handoverTtTMs;
    g_handoverHysteresisDb = handoverHysteresisDb;
    g_pingPongWindowSec = pingPongWindowSec;

    OpenKpiCsvFiles(ueKpiCsvPath, cellKpiCsvPath);

    NS_ABORT_MSG_IF(centralFrequency18 < 0.5e9 || centralFrequency18 > 100e9,
                    "Invalid 1.8 GHz NR center frequency");
    NS_ABORT_MSG_IF(centralFrequency35 < 0.5e9 || centralFrequency35 > 100e9,
                    "Invalid 3.5 GHz NR center frequency");
    NS_ABORT_MSG_IF(bandwidth18 <= 0 || bandwidth35 <= 0, "Bandwidth must be positive");
    NS_ABORT_MSG_IF(numerology > 4, "Numerology must be in [0,4]");
    NS_ABORT_MSG_IF(udpBaseIntervalMs <= 0, "UDP base interval must be positive");
    NS_ABORT_MSG_IF(pingPongWindowSec <= 0, "Ping-pong window must be positive");
    NS_ABORT_MSG_IF(slotReuseGuardSec < 0, "Slot reuse guard cannot be negative");
    NS_ABORT_MSG_IF(slotSettlingHoWindowSec < 0,
                    "Slot settling HO accounting window cannot be negative");
    NS_ABORT_MSG_IF(sionnaCacheFile18.empty() || sionnaCacheFile35.empty(),
                    "Both --sionnaCacheFile18 and --sionnaCacheFile35 are required");
    NS_ABORT_MSG_IF(sionnaCacheFile18 == sionnaCacheFile35,
                    "The 1.8 GHz and 3.5 GHz bands require distinct frequency-specific caches");

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    py::object guiClient = ConnectGuiZmqBridge(guiSrc, guiHost);
    g_guiClient = &guiClient;

    py::object influxClient = ConnectInfluxWriter(influxSrc, influxHost, influxPort, influxDb);
    g_influxClient = &influxClient;

    std::vector<UeTrace> ueTraces = LoadUeTrace(sumoTracePath, nUes);
    std::vector<GnbPosition> gnbPositions = LoadGnbPositions(gnbPositionsPath);
    ValidateSionnaCacheTopology(
        sionnaCacheFile18, gnbPositions, NrBandKind::LOW_18_GHZ, centralFrequency18);
    ValidateSionnaCacheTopology(
        sionnaCacheFile35, gnbPositions, NrBandKind::MID_35_GHZ, centralFrequency35);

    uint32_t computedPoolSize = 0;
    uint32_t maxConcurrent = 0;
    std::vector<UeSession> sessions = BuildSessionsAndAssignSlots(ueTraces,
                                                                  simTime.GetSeconds(),
                                                                  reuseUeSlots,
                                                                  slotReuseGuardSec,
                                                                  &computedPoolSize,
                                                                  &maxConcurrent);
    NS_ABORT_MSG_IF(computedPoolSize == 0, "trace produced zero active sessions within simTime");

    uint32_t poolSize = (poolSizeOverride > 0) ? poolSizeOverride : computedPoolSize;
    NS_ABORT_MSG_IF(poolSize < computedPoolSize,
                    "--poolSize=" << poolSize << " is smaller than the trace's true max "
                                  << "concurrent active count (" << computedPoolSize
                                  << "); some sessions would have nowhere to go");

    if (reuseUeSlots)
    {
        NS_LOG_UNCOND(ueTraces.size() << " distinct UE ids, " << sessions.size()
                                      << " active sessions, max concurrent = " << maxConcurrent
                                      << " -> UE device count = " << poolSize
                                      << " (reusable pool, guard=" << slotReuseGuardSec
                                      << "s, settlingHoKpiWindow="
                                      << slotSettlingHoWindowSec << "s)");
    }
    else
    {
        NS_LOG_UNCOND(ueTraces.size() << " distinct UE ids, " << sessions.size()
                                      << " active sessions, max concurrent = " << maxConcurrent
                                      << " -> UE device count = " << poolSize
                                      << " (session-isolated)");
    }
    NS_LOG_UNCOND("[handover] TTT=" << handoverTtTMs
                                     << "ms hysteresis=" << handoverHysteresisDb
                                     << "dB pingPongWindow=" << pingPongWindowSec << "s");

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

    NS_LOG_UNCOND("Creating " << ueNodes.GetN() << " UE devices and " << gnbNodes.GetN()
                              << " gNBs");

    Ptr<ListPositionAllocator> gnbPositionAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < gnbPositions.size(); ++i)
    {
        gnbPositionAlloc->Add(gnbPositions[i].position);
        NS_LOG_UNCOND("gNB '" << gnbPositions[i].externalId << "' -> gNB " << i << " pos=("
                              << gnbPositions[i].position.x << "," << gnbPositions[i].position.y
                              << "," << gnbPositions[i].position.z << ")");
        SendGnbPositionToGui(guiClient, gnbPositions[i].externalId, gnbPositions[i].position);

        const Vector cellColor = CellColorForName(gnbPositions[i].externalId, i);
        SendColorToGui(guiClient, gnbPositions[i].externalId, cellColor);
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

    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;
    CcBwpCreator::SimpleOperationBandConf bandConf18(centralFrequency18,
                                                     bandwidth18,
                                                     numCcPerBand);
    CcBwpCreator::SimpleOperationBandConf bandConf35(centralFrequency35,
                                                     bandwidth35,
                                                     numCcPerBand);
    OperationBandInfo band18 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf18);
    OperationBandInfo band35 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf35);

    Ptr<NrChannelHelper> channelHelper18 = CreateObject<NrChannelHelper>();
    channelHelper18->SetAttribute("ChannelModel", StringValue("SionnaRT"));
    channelHelper18->ConfigureSpectrumFactory(SionnaRtSpectrumPropagationLossModel::GetTypeId());
    channelHelper18->AssignChannelsToBands({band18}, NrChannelHelper::INIT_FADING);

    Ptr<NrChannelHelper> channelHelper35 = CreateObject<NrChannelHelper>();
    channelHelper35->SetAttribute("ChannelModel", StringValue("SionnaRT"));
    channelHelper35->ConfigureSpectrumFactory(SionnaRtSpectrumPropagationLossModel::GetTypeId());
    channelHelper35->AssignChannelsToBands({band35}, NrChannelHelper::INIT_FADING);

    BandwidthPartInfoPtrVector bwps18 = CcBwpCreator::GetAllBwps({band18});
    BandwidthPartInfoPtrVector bwps35 = CcBwpCreator::GetAllBwps({band35});
    ConfigureSionnaLookupCache(bwps18,
                               sionnaCacheFile18,
                               centralFrequency18,
                               sionnaUpdatePeriod,
                               "low-band NR (gNB_4G labels)");
    ConfigureSionnaLookupCache(bwps35,
                               sionnaCacheFile35,
                               centralFrequency35,
                               sionnaUpdatePeriod,
                               "mid-band NR (gNB_5G labels)");

    Packet::EnableChecking();

    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));
    nrEpcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(2));
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

    // Real RSRP/A3-event-driven handover, not the NoOp default. Pool-slot
    // reuse is handled as disconnect plus fresh attach, so every handover
    // seen here belongs to mobility of the current logical subscriber.
    nrHelper->SetHandoverAlgorithmType("ns3::NrA3RsrpHandoverAlgorithm");
    nrHelper->SetHandoverAlgorithmAttribute("TimeToTrigger",
                                            TimeValue(MilliSeconds(handoverTtTMs)));
    nrHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(handoverHysteresisDb));

    std::vector<NrBandKind> gnbBandKinds;
    NetDeviceContainer gnbNetDev;
    for (uint32_t i = 0; i < gnbNodes.GetN(); ++i)
    {
        NrBandKind bandKind = GetBandKind(gnbPositions[i].externalId);
        gnbBandKinds.push_back(bandKind);
        const BandwidthPartInfoPtrVector& gnbBwps =
            (bandKind == NrBandKind::LOW_18_GHZ) ? bwps18 : bwps35;
        gnbNetDev.Add(nrHelper->InstallGnbDevice(gnbNodes.Get(i), gnbBwps));
        Ptr<NrGnbNetDevice> installedGnb =
            DynamicCast<NrGnbNetDevice>(gnbNetDev.Get(gnbNetDev.GetN() - 1));
        const uint16_t actualCellId = installedGnb->GetCellId();
        g_cellKpi[actualCellId].name = gnbPositions[i].externalId;
        g_cellIdToColor[actualCellId] = CellColorForName(gnbPositions[i].externalId, i);
        NS_LOG_UNCOND("[band] " << gnbPositions[i].externalId << " uses "
                                << ((bandKind == NrBandKind::LOW_18_GHZ) ? centralFrequency18
                                                                        : centralFrequency35)
                                << " Hz NR, actual CellId=" << actualCellId);
    }

    // NrHelper::AttachToGnb currently registers UE PHY index i before it resolves
    // the target gNB ARFCN to the corresponding UE BWP index. Since every gNB
    // has one local BWP (i=0), make each pool slot's first serving band its UE
    // BWP 0. Later inter-frequency handovers resolve by ARFCN and can switch to
    // the other BWP normally.
    NetDeviceContainer ueNetDev;
    for (uint32_t slot = 0; slot < ueNodes.GetN(); ++slot)
    {
        const Vector initialPosition = sessionsBySlot[slot].empty()
                                           ? Vector(0, 0, 0)
                                           : sessionsBySlot[slot].front()->startPosition;
        double minDistance = std::numeric_limits<double>::infinity();
        uint32_t closestGnbIndex = 0;
        for (uint32_t g = 0; g < gnbPositions.size(); ++g)
        {
            const double distance = CalculateDistance(initialPosition, gnbPositions[g].position);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestGnbIndex = g;
            }
        }

        const bool startsOnLowBand =
            gnbBandKinds[closestGnbIndex] == NrBandKind::LOW_18_GHZ;
        BandwidthPartInfoPtrVector ueBwps = startsOnLowBand
                                               ? BandwidthPartInfoPtrVector{bwps18.front(),
                                                                            bwps35.front()}
                                               : BandwidthPartInfoPtrVector{bwps35.front(),
                                                                            bwps18.front()};
        ueNetDev.Add(nrHelper->InstallUeDevice(ueNodes.Get(slot), ueBwps));
    }

    if (gnbNodes.GetN() > 1)
    {
        nrHelper->AddX2Interface(gnbNodes);
    }

    Ptr<NrGnbNetDevice> lowBandGnb;
    Ptr<NrGnbNetDevice> midBandGnb;
    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i)
    {
        Ptr<NrGnbNetDevice> gnb = DynamicCast<NrGnbNetDevice>(gnbNetDev.Get(i));
        if (gnbBandKinds[i] == NrBandKind::LOW_18_GHZ && !lowBandGnb)
        {
            lowBandGnb = gnb;
        }
        else if (gnbBandKinds[i] == NrBandKind::MID_35_GHZ && !midBandGnb)
        {
            midBandGnb = gnb;
        }
    }
    NS_ABORT_MSG_IF(!lowBandGnb || !midBandGnb,
                    "The topology needs at least one gNB_4G and one gNB_5G label");

    const uint32_t lowArfcn = lowBandGnb->GetBwpArfcn(0);
    const uint8_t lowBandwidth = lowBandGnb->GetBwpDlBandwidth(0);
    const uint32_t midArfcn = midBandGnb->GetBwpArfcn(0);
    const uint8_t midBandwidth = midBandGnb->GetBwpDlBandwidth(0);
    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i)
    {
        Ptr<NrGnbNetDevice> gnb = DynamicCast<NrGnbNetDevice>(gnbNetDev.Get(i));
        if (gnbBandKinds[i] == NrBandKind::LOW_18_GHZ)
        {
            gnb->GetRrc()->AddNeighbourMeasFrequency(midArfcn, midBandwidth);
        }
        else
        {
            gnb->GetRrc()->AddNeighbourMeasFrequency(lowArfcn, lowBandwidth);
        }
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
            ->SetAttribute("Numerology", UintegerValue(numerology));
        const double txPowerDbm = totalTxPower;
        NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)->SetAttribute("TxPower", DoubleValue(txPowerDbm));

        const uint16_t cellId = DynamicCast<NrGnbNetDevice>(gnbNetDev.Get(i))->GetCellId();
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

    // Preserve the original per-session [2x,20x) demand distribution while
    // making the base interval configurable. The 200 ms default is exactly
    // one tenth of the packet rate produced by the original 20 ms setting.
    Ptr<UniformRandomVariable> trafficMultiplierRv = CreateObject<UniformRandomVariable>();
    trafficMultiplierRv->SetAttribute("Min", DoubleValue(2.0));
    trafficMultiplierRv->SetAttribute("Max", DoubleValue(20.0));

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
            [ueDevice, mobility, &guiClient, guiName, traceId, imsi, slot, nrHelper, gnbNetDev,
             position = session.startPosition, &slotEverAttached, slotSettlingHoWindowSec]() {
                mobility->SetPosition(position);
                SendUePositionToGui(guiClient, guiName, position);

                UeKpiState& kpi = g_ueKpi[imsi];
                const bool reusingAttachedSlot = slotEverAttached[slot];
                ++kpi.sessionGeneration;
                kpi.name = traceId;
                kpi.rsrpDbm.clear();
                kpi.hoCount = 0;
                kpi.lastHoTime = -1e9;
                kpi.lastHoSource = 0;
                kpi.lastHoTarget = 0;
                kpi.isPingPong = false;
                kpi.dlSinrDb = std::numeric_limits<double>::quiet_NaN();
                kpi.dlMcs = -1;
                kpi.suppressHoAccountingUntil =
                    reusingAttachedSlot
                        ? Simulator::Now().GetSeconds() + slotSettlingHoWindowSec
                        : -1e9;
                g_pendingHoSourceCell.erase(imsi);
                g_pendingHoSessionGeneration.erase(imsi);
                g_pendingHoAccountingSuppressed.erase(imsi);
                if (kpi.serverApp)
                {
                    kpi.lastRxPackets = kpi.serverApp->GetReceived();
                }

                if (!reusingAttachedSlot)
                {
                    double minDistance = std::numeric_limits<double>::infinity();
                    Ptr<NetDevice> closest;
                    for (uint32_t g = 0; g < gnbNetDev.GetN(); ++g)
                    {
                        Vector gnbPos = gnbNetDev.Get(g)
                                            ->GetNode()
                                            ->GetObject<MobilityModel>()
                                            ->GetPosition();
                        double d = CalculateDistance(position, gnbPos);
                        if (d < minDistance)
                        {
                            minDistance = d;
                            closest = gnbNetDev.Get(g);
                        }
                    }
                    NS_ABORT_MSG_IF(!closest, "no gNB available for pool slot " << slot);
                    slotEverAttached[slot] = true;
                    AttachSessionUe(nrHelper, ueDevice, closest, slot, traceId);
                }
                else
                {
                    // Keep the physical RRC connection. Teleporting the slot
                    // changes its measurements and lets the configured A3
                    // algorithm perform a normal handover if another cell is
                    // better. Calling AttachToGnb again on a connected UE is
                    // unsupported and was the source of failed second attaches.
                    NS_LOG_UNCOND("[pool] t=" << Simulator::Now().GetSeconds()
                                              << " slot=" << slot << " '" << traceId
                                              << "' reused; current completed cell="
                                              << kpi.completedServingCellId
                                              << ", settling HO KPI suppressed for "
                                              << slotSettlingHoWindowSec << "s");
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

        // Mark the physical slot idle as soon as this logical subscriber's
        // trace ends. The RRC object remains available for reuse, but idle
        // time must not be emitted as zero-throughput KPI rows under the old
        // subscriber's name. If another session starts at this exact time,
        // this event was scheduled first and its arrival event claims the
        // slot immediately afterwards.
        Simulator::Schedule(Seconds(session.end),
                            [ueDevice, traceId, imsi, slot, reuseUeSlots]() {
            UeKpiState& kpi = g_ueKpi[imsi];
            if (kpi.name != traceId)
            {
                return;
            }
            ++kpi.sessionGeneration;
            kpi.name.clear();
            kpi.rsrpDbm.clear();
            kpi.hoCount = 0;
            kpi.lastHoTime = -1e9;
            kpi.lastHoSource = 0;
            kpi.lastHoTarget = 0;
            kpi.isPingPong = false;
            kpi.dlSinrDb = std::numeric_limits<double>::quiet_NaN();
            kpi.dlMcs = -1;
            kpi.suppressHoAccountingUntil = -1e9;
            g_pendingHoSourceCell.erase(imsi);
            g_pendingHoSessionGeneration.erase(imsi);
            g_pendingHoAccountingSuppressed.erase(imsi);
            if (kpi.serverApp)
            {
                kpi.lastRxPackets = kpi.serverApp->GetReceived();
            }
            NS_LOG_UNCOND("[pool] t=" << Simulator::Now().GetSeconds() << " slot=" << slot
                                      << " '" << traceId << "' departed");
            if (!reuseUeSlots)
            {
                DisconnectIdlePooledUe(ueDevice, imsi, slot);
            }
        });

        // Traffic for this session, clipped to [trafficStart, simTime] exactly
        // like khu-real-nr-sionna.cc's installClient() lambda.
        double start = std::max(session.start, trafficStart.GetSeconds());
        double stop = std::min(session.end, simTime.GetSeconds());
        if (stop > start)
        {
            double multiplier = trafficMultiplierEnabled ? trafficMultiplierRv->GetValue() : 1.0;
            double intervalMs = udpBaseIntervalMs / multiplier;
            constexpr double kPacketSizeBytes = 1024.0;
            double offeredKbps = (kPacketSizeBytes * 8.0) / intervalMs;

            UdpClientHelper dlClient(ueIpIface.GetAddress(slot), dlPort);
            dlClient.SetAttribute("Interval", TimeValue(MilliSeconds(intervalMs)));
            dlClient.SetAttribute("MaxPackets", UintegerValue(1000000));
            dlClient.SetAttribute("PacketSize", UintegerValue(1024));
            ApplicationContainer clientApp = dlClient.Install(remoteHost);
            clientApp.Start(Seconds(start));
            clientApp.Stop(Seconds(stop));
            clientApps.Add(clientApp);
            NS_LOG_UNCOND("[traffic] t=" << start << " slot=" << slot << " '" << traceId
                                         << "' multiplier=" << multiplier << "x ("
                                         << offeredKbps << " kbps)");
        }
    }

    // Always schedule KPI collection (drives both the CSV row buffers and,
    // when --influxSrc was given, the live InfluxDB writes) -- CSV export
    // should work the same whether or not InfluxDB is enabled/reachable.
    Simulator::Schedule(trafficStart, &ReportKpiToInflux, ueNetDev, kpiReportInterval);

    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(ueNodes);
    Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);

    // Perf diagnostic: wall-clock timestamp at every simulated second, to
    // see whether slow spots correlate with specific events (attach/HO) or
    // are spread uniformly.
    {
        static auto wallStart = std::chrono::steady_clock::now();
        static std::function<void()> heartbeat = [&]() {
            auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                          wallStart)
                              .count();
            NS_LOG_UNCOND("[PERF] simTime=" << Simulator::Now().GetSeconds()
                                            << "s wallClock=" << elapsed << "s");
            if (Simulator::Now() + Seconds(1) <= simTime)
            {
                Simulator::Schedule(Seconds(1), heartbeat);
            }
        };
        Simulator::Schedule(Seconds(0), heartbeat);
    }

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

    PrintFinalKpiSummary();
    g_ueKpiCsv.flush();
    g_cellKpiCsv.flush();
    std::cout << "[kpi-csv] wrote " << g_ueKpiRowCount << " UE rows to '" << ueKpiCsvPath
              << "' and " << g_cellKpiRowCount << " cell rows to '" << cellKpiCsvPath << "'\n";

    Simulator::Destroy();

    const bool trafficMissing = requireTraffic && totalDownlinkRxBytes == 0;
    return trafficMissing ? 1 : 0;
}
