// Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

#ifndef NR_GNB_NET_DEVICE_H
#define NR_GNB_NET_DEVICE_H

#include "nr-fh-control.h"
#include "nr-handover-algorithm.h"
#include "nr-net-device.h"
#include "nr-rrc-sap.h"

#include "ns3/deprecated.h"
#include "ns3/nr-export.h"
#include "ns3/traced-callback.h"
#include <ns3/oran-interface.h>

namespace ns3
{

class Packet;
class PacketBurst;
class Node;
class NrGnbPhy;
class NrGnbMac;
class NrGnbRrc;
class BandwidthPartGnb;
class NrGnbComponentCarrierManager;
class BwpManagerGnb;
class NrMacScheduler;
class NrBearerStatsCalculator;

/**
 * @ingroup gnb
 * @brief The NrGnbNetDevice class
 *
 * This class represent the GNB NetDevice.
 */
class NR_EXPORT NrGnbNetDevice : public NrNetDevice
{
  public:
    static TypeId GetTypeId();

    NrGnbNetDevice();

    ~NrGnbNetDevice() override;

    Ptr<NrMacScheduler> GetScheduler(uint8_t index) const;

    Ptr<NrGnbMac> GetMac(uint8_t index) const;

    Ptr<NrGnbPhy> GetPhy(uint8_t index) const;

    Ptr<BwpManagerGnb> GetBwpManager() const;

    uint16_t GetCellId(uint8_t index) const;

    /**
     * @return the cell id
     */
    uint16_t GetCellId() const;

    /**
     * @return the BWP IDs of this gNB
     */
    std::vector<uint16_t> GetBwpIds() const;

    /**
     * @brief Set this gnb cell id
     * @param cellId the cell id
     */
    void SetCellId(uint16_t cellId);

    void SetRrc(Ptr<NrGnbRrc> rrc);

    Ptr<NrGnbRrc> GetRrc();

    void SetCcMap(const std::map<uint8_t, Ptr<BandwidthPartGnb>>& ccm);

    /**
     * @brief Get the size of the component carriers map
     * @return the number of cc that we have
     */
    uint32_t GetCcMapSize() const;

    /**
     * @brief Set the NrFhControl for this cell
     * @param nrFh The ptr to the NrFhControl
     */
    void SetNrFhControl(Ptr<NrFhControl> nrFh);

    /**
     * @brief Get the NrFhControl for this cell
     * @return the ptr to NrFhControl
     */
    Ptr<NrFhControl> GetNrFhControl();

    /**
     * @brief The gNB received a CTRL message list.
     *
     * The gNB should divide the messages to the BWP they pertain to.
     *
     * @param msgList Message list
     * @param sourceBwpArfcn BWP arfcn from which the list originated
     */
    void RouteIngoingCtrlMsgs(const std::list<Ptr<NrControlMessage>>& msgList,
                              uint32_t sourceBwpArfcn);

    /**
     * @brief Route the outgoing messages to the right BWP
     * @param msgList the list of messages
     * @param sourceBwpId the source bwp of the messages
     */
    void RouteOutgoingCtrlMsgs(const std::list<Ptr<NrControlMessage>>& msgList,
                               uint8_t sourceBwpId);

    /**
     * @brief Update the RRC configuration after installation
     *
     * This method finishes cell configuration in the RRC once PHY
     * configuration is finished.  It must be called exactly once
     * for each NrGnbNetDevice.
     *
     * After NrHelper::Install() is called on gNB nodes, either this method
     * or the NrHelper::UpdateDeviceConfigs() method (which, in turn, calls
     * this method) must be called exactly once, @b after any post-install
     * PHY configuration is done (if any), and @b before any call is made
     * (if any) to attach UEs to gNBs, such as AttachToGnb() and
     * AttachToClosestGnb().
     *
     * This method will assert if called twice on the same device.
     *
     * This method is deprecated and no longer needed and will be removed
     * from future versions of this model.  It is replaced by ConfigureCell().
     */
    NS_DEPRECATED("Obsolete method")
    void UpdateConfig();

    /**
     * @brief Update the RRC configuration after installation
     *
     * This method calls ConfigureCell() on the RRC using the component
     * carrier map that has already been installed on this net device.
     *
     * This method finishes cell configuration in the RRC once PHY
     * configuration is finished.  It must be called exactly once
     * for each NrGnbNetDevice.
     *
     * After NrHelper::Install() is called on gNB nodes, either this method
     * or the NrHelper::AttachToGnb() method (or AttachToClosestGnb() method),
     * which, in turn, calls this method, must be called exactly once,
     * @b after any post-install PHY configuration is done (if any).
     *
     * If AttachToGnb() is not called by initialization time, this
     * method will be called by DoInitialize().
     *
     * This method will assert if called twice on the same device.  Users
     * may check whether it has been called already by calling the
     * IsCellConfigured() method.
     */
    void ConfigureCell();

    /**
     * @brief Return true if ConfigureCell() has been called
     * @return whether ConfigureCell() has been called
     */
    bool IsCellConfigured() const;

    /**
     * @brief Get downlink bandwidth for a given bandwidth part id
     * @param bwpId Bandwidth part Id
     * @return number of RBs
     */
    uint16_t GetBwpDlBandwidth(uint16_t bwpId) const;

    /**
     * @brief Get uplink bandwidth for a given bandwidth part id
     * @param bwpId Bandwidth part Id
     * @return number of RBs
     */
    uint16_t GetBwpUlBandwidth(uint16_t bwpId) const;

    /**
     * @brief Get earfcn for a given bandwidth part id
     * @param bwpId Bandwidth part Id
     * @return earfcn
     */
    uint32_t GetBwpArfcn(uint16_t bwpId) const;

    /**
     * @brief Get the local bandwidth part id for a target arfcn
     * @param arfcn target ARFCN of BWP
     * @return Bandwidth part Id
     */
    uint16_t GetArfcnBwpId(uint32_t arfcn) const;

    void SetE2Termination(Ptr<E2Termination> e2term);
    Ptr<E2Termination> GetE2Termination() const;
    void KpmSubscriptionCallback(E2AP_PDU_t* sub_req_pdu);
    void ControlMessageReceivedCallback(E2AP_PDU_t* sub_req_pdu);
    void stopSendingAndCancelSchedule();

    /**
     * Build one round of KPM indication messages (CU-UP and CU-CP) and send
     * them to the RIC through the E2 termination, then reschedule itself
     * after E2Periodicity seconds for as long as the subscription is alive.
     * Ported from ns3-o-ran-e2's MmWaveEnbNetDevice::BuildAndSendReportMessage.
     *
     * @param params the RIC subscription identifiers to echo in the indication
     */
    void BuildAndSendReportMessage(E2Termination::RicSubscriptionRequest_rval_s params);

    /**
     * Callback for NrGnbRrc's "RecvMeasurementReport" trace source, connected
     * to every gNB's RRC in DoInitialize when E2/CU-CP reporting is enabled.
     * Records the serving-cell and neighbour-cell RSRP (there is no L3 SINR
     * in the standard 3GPP RRC measurement report the way the legacy
     * mmwave+LteEnbRrc dual-connectivity architecture exposed one directly)
     * for later use in BuildRicIndicationMessageCuCp. Only fires for UEs
     * whose measurement configuration was actually set up -- e.g. by
     * installing an RSRP-based handover algorithm via
     * NrHelper::SetHandoverAlgorithmType (the default NrNoOpHandoverAlgorithm
     * configures no measurements at all).
     *
     * @param imsi the reporting UE's IMSI
     * @param cellId the cell the report was received on
     * @param rnti the reporting UE's RNTI
     * @param report the decoded RRC measurement report
     */
    void RecvMeasurementReport(uint64_t imsi,
                              uint16_t cellId,
                              uint16_t rnti,
                              NrRrcSap::MeasurementReport report);

    bool m_forceE2FileLogging;

  protected:
    void DoInitialize() override;

    void DoDispose() override;
    bool DoSend(Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber) override;

  private:
    /**
     * Build the E2SM-KPM indication header (PLMN, gNB id, cell id, timestamp).
     * @return the encoded header, or nullptr when offline file logging is on
     */
    Ptr<KpmIndicationHeader> BuildRicIndicationHeader(std::string plmId,
                                                      std::string gnbId,
                                                      uint16_t nrCellId);
    /**
     * Build the CU-UP indication message: per-UE PDCP/RLC downlink volume and
     * PDU counts read from the E2PdcpCalculator/E2RlcCalculator attributes.
     * @return the encoded message, or nullptr if stats calculators are absent
     */
    Ptr<KpmIndicationMessage> BuildRicIndicationMessageCuUp(std::string plmId);
    /**
     * Build the CU-CP indication message: number of active UEs and per-UE DRB
     * counts. L3 serving/neighbour SINR values are not plumbed from the NR RRC
     * measurement path yet and are reported as 0.
     * @return the encoded message
     */
    Ptr<KpmIndicationMessage> BuildRicIndicationMessageCuCp(std::string plmId);
    /**
     * Zero-pad an IMSI to the 5-character string format the E2SM-KPM UE id
     * field expects.
     * @return the padded IMSI string
     */
    static std::string GetImsiString(uint64_t imsi);
    /**
     * Apply a RET (Remote Electrical Tilt) control decision to every BWP
     * antenna of this cell by setting the UniformPlanarArray BearingAngle /
     * DowntiltAngle attributes. With the Sionna RT channel model these
     * orientations are forwarded to the ray tracer on the next channel
     * update, so the tilt has a real propagation effect.
     */
    void ApplyRetControl(double tiltDeg, bool hasBearing, double bearingDeg);

    Ptr<NrGnbRrc> m_rrc;
    Ptr<NrHandoverAlgorithm> m_handoverAlgorithm; ///< the handover algorithm

    uint16_t m_cellId; //!< Cell ID. Set by the helper.

    std::map<uint8_t, Ptr<BandwidthPartGnb>> m_ccMap; /**< NrComponentCarrier map */

    Ptr<NrGnbComponentCarrierManager>
        m_componentCarrierManager; ///< the component carrier manager of this gNB
    Ptr<NrFhControl> m_nrFhControl;

    bool m_isCellConfigured{false}; ///< variable to check whether the RRC has been configured

    Ptr<E2Termination> m_e2term;
    Ptr<NrBearerStatsCalculator> m_e2PdcpStatsCalculator; //!< PDCP stats source for KPM reports
    Ptr<NrBearerStatsCalculator> m_e2RlcStatsCalculator;  //!< RLC stats source for KPM reports
    double rc_e2_func_id;  //!< RC function ID
    double e2_func_id;     //!< KPM function ID
    double m_e2Periodicity; //!< KPM indication period in seconds
    bool m_sendCuUp;        //!< send the CU-UP indication message
    bool m_sendCuCp;        //!< send the CU-CP indication message
    bool m_reducedPmValues; //!< use the reduced PM value set in indications
    bool m_stopSendingMessages;
    bool m_isReportingEnabled;
    bool m_hasValidSubscription{false}; //!< a RIC subscription has been processed
    E2Termination::RicSubscriptionRequest_rval_s
        m_lastSubscriptionParams; //!< identifiers of the last RIC subscription
    uint64_t m_startTime{0};      //!< epoch offset (ms) added to indication timestamps
    /**
     * Last-known RSRP (dBm) per (IMSI, cellId) from RRC measurement reports,
     * covering both the serving cell and any reported neighbours. Populated
     * by RecvMeasurementReport, consumed by BuildRicIndicationMessageCuCp.
     */
    std::map<uint64_t, std::map<uint16_t, double>> m_l3RsrpDbmMap;
};

} // namespace ns3

#endif /* NR_GNB_NET_DEVICE_H */
