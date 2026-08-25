/*
 * SPDX-License-Identifier: GPL-2.0-only
 */
#ifndef SIONNA_LOOKUP_CHANNEL_MODEL_H
#define SIONNA_LOOKUP_CHANNEL_MODEL_H

#include "matrix-based-channel-model.h"

#include "ns3/vector.h"

#include <complex.h>
#include <map>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

namespace ns3
{

class MobilityModel;

/**
 * @ingroup spectrum
 * @brief Drop-in replacement for SionnaRtChannelModel that answers
 * GetChannel()/GetParams() from a precomputed HDF5 lookup table instead of
 * invoking the Sionna RT PathSolver live.
 *
 * The cache is built offline by
 * scenarios/khu-real/tools/build_sionna_rt_cache.py, which runs the exact
 * same PathSolver call (same antenna arrays, same RtPathSolverConfig, same
 * bearing/tilt) that SionnaRtChannelModel would at runtime, for every gNB in
 * gnbs-ret.csv against every distinct UE position seen across the
 * ue_positions_seed*.csv traces (rounded to a 1m x/y, 0.1m z grid). At
 * runtime this class identifies which endpoint of a GetChannel() call is a
 * cached gNB by matching its (fixed) position against the cache's stored
 * gNB positions, snaps the other endpoint's position to the same grid, and
 * looks the precomputed channel straight up -- no ray tracing, no Python
 * scene/PathSolver calls per query.
 *
 * Caveats inherent to this approach (by design, not oversights):
 *  - Only valid for the fixed bearing/tilt baked into the cache at build
 *    time (see gnbs-ret.csv at cache-build time). A live RET tilt/bearing
 *    change (NrGnbNetDevice::ApplyRetControl) is NOT reflected here -- the
 *    cache would need to be rebuilt per tilt value swept, which is a
 *    separate, larger feature (see conversation notes on tilt-sweep cache
 *    sizing) not implemented by this class.
 *  - No Doppler modeling: ChannelParams::m_alpha/m_D are filled with zeros
 *    (no per-cluster Doppler), consistent with treating each grid cell as a
 *    static snapshot -- the same approximation SionnaRtChannelModel's own
 *    UpdatePeriod staleness window already makes between live recomputes.
 *  - A query position that isn't an exact grid hit (e.g. a UE trace not
 *    among the ones the cache was built from) falls back to a brute-force
 *    nearest-neighbor search among that gNB's ~12,660 cached positions --
 *    correct, but not O(1) like the common case.
 */
class SionnaLookupChannelModel : public MatrixBasedChannelModel
{
  public:
    SionnaLookupChannelModel();
    ~SionnaLookupChannelModel() override;

    void DoDispose() override;

    static TypeId GetTypeId();

    void SetCacheFile(const std::string& path);
    std::string GetCacheFile() const;

    void SetFrequency(double f);
    double GetFrequency() const;

    Ptr<const ChannelMatrix> GetChannel(Ptr<const MobilityModel> aMob,
                                        Ptr<const MobilityModel> bMob,
                                        Ptr<const PhasedArrayModel> aAntenna,
                                        Ptr<const PhasedArrayModel> bAntenna) override;

    Ptr<const ChannelParams> GetParams(Ptr<const MobilityModel> aMob,
                                       Ptr<const MobilityModel> bMob) const override;

  private:
    /// One gNB's worth of cached data, loaded once from the HDF5 file.
    struct GnbCacheEntry
    {
        Vector position;
        std::vector<Vector> positions; //!< query positions, row-aligned with the arrays below
        std::map<std::tuple<int, int, int>, size_t> keyToIndex; //!< (round(x),round(y),round(z*10)) -> row

        uint32_t numRxAnt = 0;
        uint32_t numTxAnt = 0;
        uint32_t maxPaths = 0;

        // Flattened [row][rxAnt][txAnt][path] and [row][path] arrays. Kept
        // at their on-disk precision (complex64/float32, not double) --
        // this is 3 gNBs x 12660 positions x 8x32 antennas x 128 padded
        // paths, so upcasting to double here (as an earlier version of this
        // code did, both on the numpy-read side and the C++ storage side)
        // peaks at ~30GB of transient memory and OOM-kills the process.
        std::vector<std::complex<float>> a; // row-major over (row, rxAnt, txAnt, path)
        std::vector<float> tau;             // same layout as a
        std::vector<float> thetaT;          // row-major over (row, path)
        std::vector<float> phiT;
        std::vector<float> thetaR;
        std::vector<float> phiR;
        std::vector<uint32_t> numPaths; //!< actual (unpadded) path count per row
    };

    void EnsureCacheLoaded() const;
    void LoadCache() const;

    /// Returns the loaded gNB entry whose stored position matches `pos`
    /// within 1cm, or nullptr if `pos` isn't one of the cached gNBs.
    const GnbCacheEntry* FindGnbByPosition(const Vector& pos) const;

    /// Row index into a GnbCacheEntry's arrays for the (possibly
    /// off-grid) query position, snapping to the same 1m/1m/0.1m grid the
    /// cache was built with, falling back to nearest-neighbor on a miss.
    size_t FindRow(const GnbCacheEntry& entry, const Vector& queryPos) const;

    Ptr<ChannelMatrix> BuildChannelMatrix(const GnbCacheEntry& gnb,
                                          size_t row,
                                          const Ptr<const MobilityModel>& aMob,
                                          const Ptr<const MobilityModel>& bMob,
                                          Ptr<const PhasedArrayModel> aAntenna,
                                          Ptr<const PhasedArrayModel> bAntenna,
                                          bool gnbIsA) const;

    Ptr<ChannelParams> BuildChannelParams(const GnbCacheEntry& gnb,
                                          size_t row,
                                          const Ptr<const MobilityModel>& aMob,
                                          const Ptr<const MobilityModel>& bMob) const;

    mutable std::map<std::string, GnbCacheEntry> m_gnbCache; //!< keyed by HDF5 group (gNB) name
    mutable bool m_cacheLoaded = false;
    std::string m_cacheFile;
    double m_frequency = 3.5e9;

    mutable std::unordered_map<uint64_t, Ptr<ChannelMatrix>> m_channelMatrixMap;
    mutable std::unordered_map<uint64_t, Ptr<ChannelParams>> m_channelParamsMap;
};

} // namespace ns3

#endif // SIONNA_LOOKUP_CHANNEL_MODEL_H
