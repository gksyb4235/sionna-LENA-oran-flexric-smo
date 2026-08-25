/*
 * SPDX-License-Identifier: GPL-2.0-only
 */
#include "sionna-lookup-channel-model.h"

#include "ns3/abort.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/node.h"
#include "ns3/phased-array-model.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"

#include <cmath>
#include <limits>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SionnaLookupChannelModel");
NS_OBJECT_ENSURE_REGISTERED(SionnaLookupChannelModel);

TypeId
SionnaLookupChannelModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SionnaLookupChannelModel")
            .SetParent<MatrixBasedChannelModel>()
            .SetGroupName("Spectrum")
            .AddConstructor<SionnaLookupChannelModel>()
            .AddAttribute("CacheFile",
                          "Path to the HDF5 file built by "
                          "scenarios/khu-real/tools/build_sionna_rt_cache.py",
                          StringValue(""),
                          MakeStringAccessor(&SionnaLookupChannelModel::SetCacheFile,
                                             &SionnaLookupChannelModel::GetCacheFile),
                          MakeStringChecker())
            .AddAttribute("Frequency",
                          "Center frequency in Hz (must match the cache's build-time frequency)",
                          DoubleValue(3.5e9),
                          MakeDoubleAccessor(&SionnaLookupChannelModel::SetFrequency,
                                             &SionnaLookupChannelModel::GetFrequency),
                          MakeDoubleChecker<double>());
    return tid;
}

SionnaLookupChannelModel::SionnaLookupChannelModel()
{
    NS_LOG_FUNCTION(this);
}

SionnaLookupChannelModel::~SionnaLookupChannelModel()
{
    NS_LOG_FUNCTION(this);
}

void
SionnaLookupChannelModel::DoDispose()
{
    m_channelMatrixMap.clear();
    m_channelParamsMap.clear();
    m_gnbCache.clear();
}

void
SionnaLookupChannelModel::SetCacheFile(const std::string& path)
{
    m_cacheFile = path;
}

std::string
SionnaLookupChannelModel::GetCacheFile() const
{
    return m_cacheFile;
}

void
SionnaLookupChannelModel::SetFrequency(double f)
{
    m_frequency = f;
}

double
SionnaLookupChannelModel::GetFrequency() const
{
    return m_frequency;
}

void
SionnaLookupChannelModel::EnsureCacheLoaded() const
{
    if (!m_cacheLoaded)
    {
        LoadCache();
        m_cacheLoaded = true;
    }
}

void
SionnaLookupChannelModel::LoadCache() const
{
    NS_ABORT_MSG_IF(m_cacheFile.empty(),
                    "SionnaLookupChannelModel::CacheFile must be set to the HDF5 file built by "
                    "build_sionna_rt_cache.py");
    NS_LOG_UNCOND("[SionnaLookup] loading cache from " << m_cacheFile);

    py::module_ h5py = py::module_::import("h5py");
    py::module_ np = py::module_::import("numpy");
    py::object hf = h5py.attr("File")(m_cacheFile, "r");

    for (auto item : hf.attr("keys")())
    {
        std::string gnbName = py::str(item);
        py::object group = hf[py::str(gnbName)];

        GnbCacheEntry entry;
        // Deliberately no dtype= cast here (np.asarray(x) keeps x's native
        // on-disk dtype -- complex64/float32) -- forcing float64/complex128
        // doubles every array's size *and* leaves both the upcast numpy
        // array and the eventual C++ copy alive simultaneously, which is
        // what OOM-killed this process the first time (peaked around 30GB
        // for what should be a ~2.5GB-on-disk cache).
        py::object gnbPosArr = np.attr("asarray")(group.attr("attrs")["gnb_position"]);
        auto gnbPosBuf = py::buffer(gnbPosArr).request();
        auto* gnbPosData = static_cast<double*>(gnbPosBuf.ptr);
        entry.position = Vector(gnbPosData[0], gnbPosData[1], gnbPosData[2]);

        py::object posArr = np.attr("asarray")(group["positions"]);
        auto posBuf = py::buffer(posArr).request();
        size_t n = posBuf.shape[0];
        auto* posData = static_cast<double*>(posBuf.ptr);
        entry.positions.resize(n);
        for (size_t i = 0; i < n; ++i)
        {
            double x = posData[i * 3 + 0];
            double y = posData[i * 3 + 1];
            double z = posData[i * 3 + 2];
            entry.positions[i] = Vector(x, y, z);
            auto key = std::make_tuple(static_cast<int>(std::lround(x)),
                                       static_cast<int>(std::lround(y)),
                                       static_cast<int>(std::lround(z * 10.0)));
            entry.keyToIndex[key] = i;
        }

        py::object aArr = np.attr("asarray")(group["a"]); // complex64
        auto aBuf = py::buffer(aArr).request();
        // shape: [n, numRxAnt, numTxAnt, maxPaths]
        entry.numRxAnt = static_cast<uint32_t>(aBuf.shape[1]);
        entry.numTxAnt = static_cast<uint32_t>(aBuf.shape[2]);
        entry.maxPaths = static_cast<uint32_t>(aBuf.shape[3]);
        size_t aCount = static_cast<size_t>(aBuf.size);
        auto* aData = static_cast<std::complex<float>*>(aBuf.ptr);
        entry.a.assign(aData, aData + aCount);
        aArr = py::none(); // drop the numpy-side copy before loading the next array

        py::object tauArr = np.attr("asarray")(group["tau"]); // float32
        auto tauBuf = py::buffer(tauArr).request();
        auto* tauData = static_cast<float*>(tauBuf.ptr);
        entry.tau.assign(tauData, tauData + tauBuf.size);
        tauArr = py::none();

        auto loadAngle = [&](const char* dsName) {
            py::object arr = np.attr("asarray")(group[dsName]); // float32
            auto buf = py::buffer(arr).request();
            auto* data = static_cast<float*>(buf.ptr);
            return std::vector<float>(data, data + buf.size);
        };
        entry.thetaT = loadAngle("theta_t");
        entry.phiT = loadAngle("phi_t");
        entry.thetaR = loadAngle("theta_r");
        entry.phiR = loadAngle("phi_r");

        py::object numPathsArr =
            np.attr("asarray")(group["num_paths"], py::arg("dtype") = np.attr("int64"));
        auto npBuf = py::buffer(numPathsArr).request();
        auto* npData = static_cast<int64_t*>(npBuf.ptr);
        entry.numPaths.resize(n);
        for (size_t i = 0; i < n; ++i)
        {
            entry.numPaths[i] = static_cast<uint32_t>(npData[i]);
        }

        NS_LOG_UNCOND("[SionnaLookup] loaded '" << gnbName << "': " << n << " positions, "
                                                << entry.numRxAnt << "x" << entry.numTxAnt
                                                << " antennas, maxPaths=" << entry.maxPaths);
        m_gnbCache[gnbName] = std::move(entry);
    }

    NS_ABORT_MSG_IF(m_gnbCache.empty(), "SionnaLookupChannelModel: cache file has no gNB groups: "
                                            << m_cacheFile);
}

const SionnaLookupChannelModel::GnbCacheEntry*
SionnaLookupChannelModel::FindGnbByPosition(const Vector& pos) const
{
    constexpr double kTolerance = 0.05; // meters
    for (const auto& [name, entry] : m_gnbCache)
    {
        double d = CalculateDistance(pos, entry.position);
        if (d < kTolerance)
        {
            return &entry;
        }
    }
    return nullptr;
}

size_t
SionnaLookupChannelModel::FindRow(const GnbCacheEntry& entry, const Vector& queryPos) const
{
    auto key = std::make_tuple(static_cast<int>(std::lround(queryPos.x)),
                               static_cast<int>(std::lround(queryPos.y)),
                               static_cast<int>(std::lround(queryPos.z * 10.0)));
    if (auto it = entry.keyToIndex.find(key); it != entry.keyToIndex.end())
    {
        return it->second;
    }

    // Off-grid (e.g. a UE trace not among the ones the cache was built
    // from): fall back to brute-force nearest neighbor. Expected to be rare
    // for the seed0-29 traces this cache targets.
    NS_LOG_WARN("[SionnaLookup] position (" << queryPos.x << "," << queryPos.y << "," << queryPos.z
                                            << ") is off the cache grid; falling back to nearest "
                                               "neighbor");
    size_t best = 0;
    double bestDist = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < entry.positions.size(); ++i)
    {
        double d = CalculateDistance(queryPos, entry.positions[i]);
        if (d < bestDist)
        {
            bestDist = d;
            best = i;
        }
    }
    return best;
}

Ptr<MatrixBasedChannelModel::ChannelMatrix>
SionnaLookupChannelModel::BuildChannelMatrix(const GnbCacheEntry& gnb,
                                             size_t row,
                                             const Ptr<const MobilityModel>& aMob,
                                             const Ptr<const MobilityModel>& bMob,
                                             Ptr<const PhasedArrayModel> aAntenna,
                                             Ptr<const PhasedArrayModel> bAntenna,
                                             bool gnbIsA) const
{
    uint32_t numPaths = gnb.numPaths[row];
    // The cache is always stored as [rx=UE][tx=gNB][path] (that's how the
    // offline PathSolver call in build_sionna_rt_cache.py was set up). But
    // SionnaRtSpectrumPropagationLossModel::CalcLongTerm asserts the *live
    // caller's* "b" antenna element count against m_channel's row count and
    // "a" against column count -- and the caller's a/b are literally
    // whichever node is transmitting vs receiving *that packet*, which
    // flips between downlink (a=gNB) and uplink (a=UE) for the same node
    // pair. So orientation here must follow the caller's (a,b) order, not a
    // fixed gNB/UE identity -- transpose the cached data when the caller
    // order is the reverse of how it was cached (channel reciprocity,
    // H_ab = H_ba^T, makes this transpose physically valid, matching this
    // class's -- and the base class's -- own documented contract).
    // a=gNB(tx),b=UE(rx): rows=b=UE=numRxAnt, cols=a=gNB=numTxAnt (cache-native shape).
    // a=UE(tx),b=gNB(rx): rows=b=gNB=numTxAnt, cols=a=UE=numRxAnt (transposed shape).
    Complex3DVector hUsn(gnbIsA ? gnb.numRxAnt : gnb.numTxAnt,
                         gnbIsA ? gnb.numTxAnt : gnb.numRxAnt,
                         numPaths);
    size_t rowBase = row * gnb.numRxAnt * gnb.numTxAnt * gnb.maxPaths;
    for (size_t ueIdx = 0; ueIdx < gnb.numRxAnt; ++ueIdx)
    {
        for (size_t gnbIdx = 0; gnbIdx < gnb.numTxAnt; ++gnbIdx)
        {
            size_t base = rowBase + (ueIdx * gnb.numTxAnt + gnbIdx) * gnb.maxPaths;
            for (uint32_t p = 0; p < numPaths; ++p)
            {
                std::complex<double> val = gnb.a[base + p];
                if (gnbIsA)
                {
                    // a=gNB(tx), b=UE(rx): rows=b=UE, cols=a=gNB -- matches
                    // the cache's own native layout directly.
                    hUsn(ueIdx, gnbIdx, p) = val;
                }
                else
                {
                    // a=UE(tx), b=gNB(rx): rows=b=gNB, cols=a=UE -- transpose.
                    hUsn(gnbIdx, ueIdx, p) = val;
                }
            }
        }
    }

    Ptr<ChannelMatrix> channelMatrix = Create<ChannelMatrix>();
    channelMatrix->m_generatedTime = Simulator::Now();
    channelMatrix->m_channel = hUsn;
    channelMatrix->m_nodeIds =
        std::make_pair(aMob->GetObject<Node>()->GetId(), bMob->GetObject<Node>()->GetId());
    channelMatrix->m_antennaPair = std::make_pair(aAntenna->GetId(), bAntenna->GetId());
    return channelMatrix;
}

Ptr<MatrixBasedChannelModel::ChannelParams>
SionnaLookupChannelModel::BuildChannelParams(const GnbCacheEntry& gnb,
                                             size_t row,
                                             const Ptr<const MobilityModel>& aMob,
                                             const Ptr<const MobilityModel>& bMob) const
{
    uint32_t numPaths = gnb.numPaths[row];
    Ptr<ChannelParams> params = Create<ChannelParams>();
    params->m_generatedTime = Simulator::Now();
    params->m_nodeIds =
        std::make_pair(aMob->GetObject<Node>()->GetId(), bMob->GetObject<Node>()->GetId());

    size_t angleBase = row * gnb.maxPaths;
    params->m_delay.resize(numPaths);
    params->m_angle.assign(4, DoubleVector(numPaths));
    params->m_alpha.assign(numPaths, 0.0); // no Doppler modeling, see class doc
    params->m_D.assign(numPaths, 0.0);
    for (uint32_t p = 0; p < numPaths; ++p)
    {
        // tau is stored per (rxAnt=0, txAnt=0, path) as a representative,
        // antenna-independent delay -- see build_sionna_rt_cache.py.
        params->m_delay[p] = gnb.tau[row * gnb.numRxAnt * gnb.numTxAnt * gnb.maxPaths + p];
        double aoaDeg = gnb.phiR[angleBase + p] * 180.0 / M_PI;
        double zoaDeg = gnb.thetaR[angleBase + p] * 180.0 / M_PI;
        double aodDeg = gnb.phiT[angleBase + p] * 180.0 / M_PI;
        double zodDeg = gnb.thetaT[angleBase + p] * 180.0 / M_PI;
        params->m_angle[AOA_INDEX][p] = aoaDeg;
        params->m_angle[ZOA_INDEX][p] = zoaDeg;
        params->m_angle[AOD_INDEX][p] = aodDeg;
        params->m_angle[ZOD_INDEX][p] = zodDeg;
    }

    params->m_cachedAngleSincos.resize(4);
    for (size_t direction = 0; direction < 4; ++direction)
    {
        params->m_cachedAngleSincos[direction].resize(numPaths);
        for (uint32_t p = 0; p < numPaths; ++p)
        {
            double rad = params->m_angle[direction][p] * M_PI / 180.0;
            params->m_cachedAngleSincos[direction][p] = {std::sin(rad), std::cos(rad)};
        }
    }
    return params;
}

Ptr<const MatrixBasedChannelModel::ChannelMatrix>
SionnaLookupChannelModel::GetChannel(Ptr<const MobilityModel> aMob,
                                     Ptr<const MobilityModel> bMob,
                                     Ptr<const PhasedArrayModel> aAntenna,
                                     Ptr<const PhasedArrayModel> bAntenna)
{
    EnsureCacheLoaded();

    uint64_t key = GetKey(aAntenna->GetId(), bAntenna->GetId());
    if (auto it = m_channelMatrixMap.find(key); it != m_channelMatrixMap.end())
    {
        return it->second;
    }

    const GnbCacheEntry* gnbEntry = FindGnbByPosition(aMob->GetPosition());
    bool gnbIsA = (gnbEntry != nullptr);
    if (!gnbEntry)
    {
        gnbEntry = FindGnbByPosition(bMob->GetPosition());
    }
    NS_ABORT_MSG_IF(!gnbEntry,
                    "SionnaLookupChannelModel: neither endpoint matches a cached gNB position "
                    "-- is gnbs-ret.csv different from the one the cache was built with?");

    Ptr<const MobilityModel> ueMob = gnbIsA ? bMob : aMob;

    size_t row = FindRow(*gnbEntry, ueMob->GetPosition());
    Ptr<ChannelMatrix> channelMatrix =
        BuildChannelMatrix(*gnbEntry, row, aMob, bMob, aAntenna, bAntenna, gnbIsA);
    m_channelMatrixMap[key] = channelMatrix;
    NS_LOG_UNCOND("[SionnaLookup] DEBUG GetChannel gnbIsA="
                  << gnbIsA << " aAntennaId=" << aAntenna->GetId()
                  << " bAntennaId=" << bAntenna->GetId() << " aPorts=" << aAntenna->GetNumPorts()
                  << " bPorts=" << bAntenna->GetNumPorts() << " matrixRows="
                  << channelMatrix->m_channel.GetNumRows()
                  << " matrixCols=" << channelMatrix->m_channel.GetNumCols());

    uint64_t paramsKey = GetKey(aMob->GetObject<Node>()->GetId(), bMob->GetObject<Node>()->GetId());
    m_channelParamsMap[paramsKey] = BuildChannelParams(*gnbEntry, row, aMob, bMob);

    return channelMatrix;
}

Ptr<const MatrixBasedChannelModel::ChannelParams>
SionnaLookupChannelModel::GetParams(Ptr<const MobilityModel> aMob,
                                    Ptr<const MobilityModel> bMob) const
{
    EnsureCacheLoaded();
    uint64_t key = GetKey(aMob->GetObject<Node>()->GetId(), bMob->GetObject<Node>()->GetId());
    if (auto it = m_channelParamsMap.find(key); it != m_channelParamsMap.end())
    {
        return it->second;
    }

    // Not computed yet via GetChannel (shouldn't normally happen -- NR always
    // calls GetChannel first -- but build it directly so GetParams works
    // standalone too, mirroring SionnaRtChannelModel's own GetParams, which
    // only ever reads a map GetChannel already populated).
    const GnbCacheEntry* gnbEntry = FindGnbByPosition(aMob->GetPosition());
    if (!gnbEntry)
    {
        gnbEntry = FindGnbByPosition(bMob->GetPosition());
    }
    if (!gnbEntry)
    {
        NS_LOG_WARN("[SionnaLookup] GetParams: neither endpoint matches a cached gNB; "
                    "returning nullptr");
        return nullptr;
    }
    Ptr<const MobilityModel> ueMob =
        (CalculateDistance(aMob->GetPosition(), gnbEntry->position) < 0.05) ? bMob : aMob;
    size_t row = FindRow(*gnbEntry, ueMob->GetPosition());
    Ptr<ChannelParams> params = BuildChannelParams(*gnbEntry, row, aMob, bMob);
    m_channelParamsMap[key] = params;
    return params;
}

} // namespace ns3
