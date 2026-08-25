// Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

#include "nr-gnb-net-device.h"

#include "bandwidth-part-gnb.h"
#include "bwp-manager-gnb.h"
#include "nr-gnb-component-carrier-manager.h"
#include "nr-gnb-mac.h"
#include "nr-gnb-phy.h"
#include "nr-gnb-rrc.h"
#include "nr-phy-mac-common.h"
#include "nr-radio-bearer-info.h"
#include "nr-spectrum-phy.h"

#include "ns3/abort.h"
#include "ns3/config.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/log.h"
#include "ns3/mmwave-indication-message-helper.h"
#include "ns3/nr-bearer-stats-calculator.h"
#include "ns3/object-map.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"
#include "ns3/uniform-planar-array.h"

#include <encode_e2apv1.hpp>

#include <unordered_map>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NrGnbNetDevice");

NS_OBJECT_ENSURE_REGISTERED(NrGnbNetDevice);

TypeId
NrGnbNetDevice::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::NrGnbNetDevice")
            .SetParent<NrNetDevice>()
            .AddConstructor<NrGnbNetDevice>()
            .AddAttribute("NrGnbComponentCarrierManager",
                          "The component carrier manager associated to this GnbNetDevice",
                          PointerValue(),
                          MakePointerAccessor(&NrGnbNetDevice::m_componentCarrierManager),
                          MakePointerChecker<NrGnbComponentCarrierManager>())
            .AddAttribute("BandwidthPartMap",
                          "List of Bandwidth Part container.",
                          ObjectMapValue(),
                          MakeObjectMapAccessor(&NrGnbNetDevice::m_ccMap),
                          MakeObjectMapChecker<BandwidthPartGnb>())
            .AddAttribute("NrGnbRrc",
                          "The RRC layer associated with the gNB",
                          PointerValue(),
                          MakePointerAccessor(&NrGnbNetDevice::m_rrc),
                          MakePointerChecker<NrGnbRrc>())
            .AddAttribute("NrHandoverAlgorithm",
                          "The handover algorithm associated to this GnbNetDevice",
                          PointerValue(),
                          MakePointerAccessor(&NrGnbNetDevice::m_handoverAlgorithm),
                          MakePointerChecker<NrHandoverAlgorithm>())
            .AddAttribute("E2Termination",
                          "The E2 termination object associated to this node",
                          PointerValue(),
                          MakePointerAccessor(&NrGnbNetDevice::SetE2Termination,
                                             &NrGnbNetDevice::GetE2Termination),
                          MakePointerChecker<E2Termination>())
            .AddAttribute("EnableE2FileLogging",
                          "If true, force E2 indication generation and write E2 fields in csv file",
                          BooleanValue(false),
                          MakeBooleanAccessor(&NrGnbNetDevice::m_forceE2FileLogging),
                          MakeBooleanChecker())
            .AddAttribute("KPM_E2functionID",
                          "Function ID to subscribe",
                          DoubleValue(2),
                          MakeDoubleAccessor(&NrGnbNetDevice::e2_func_id),
                          MakeDoubleChecker<double>())
            .AddAttribute("RC_E2functionID",
                          "Function ID to subscribe",
                          DoubleValue(3),
                          MakeDoubleAccessor(&NrGnbNetDevice::rc_e2_func_id),
                          MakeDoubleChecker<double>())
            .AddAttribute("E2Periodicity",
                          "Periodicity of E2 KPM indication reports (seconds)",
                          DoubleValue(1.0),
                          MakeDoubleAccessor(&NrGnbNetDevice::m_e2Periodicity),
                          MakeDoubleChecker<double>())
            .AddAttribute("E2PdcpCalculator",
                          "The PDCP stats calculator used to fill KPM CU-UP indications",
                          PointerValue(),
                          MakePointerAccessor(&NrGnbNetDevice::m_e2PdcpStatsCalculator),
                          MakePointerChecker<NrBearerStatsCalculator>())
            .AddAttribute("E2RlcCalculator",
                          "The RLC stats calculator used to fill KPM CU-UP indications",
                          PointerValue(),
                          MakePointerAccessor(&NrGnbNetDevice::m_e2RlcStatsCalculator),
                          MakePointerChecker<NrBearerStatsCalculator>())
            .AddAttribute("EnableCuUpReport",
                          "If true, send the KPM CU-UP indication message",
                          BooleanValue(true),
                          MakeBooleanAccessor(&NrGnbNetDevice::m_sendCuUp),
                          MakeBooleanChecker())
            .AddAttribute("EnableCuCpReport",
                          "If true, send the KPM CU-CP indication message",
                          BooleanValue(true),
                          MakeBooleanAccessor(&NrGnbNetDevice::m_sendCuCp),
                          MakeBooleanChecker())
            .AddAttribute("EnableDuReport",
                          "If true, send the KPM DU indication message (PRB "
                          "utilization / MCS distribution)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&NrGnbNetDevice::m_sendDu),
                          MakeBooleanChecker())
            .AddAttribute("ReducedPmValues",
                          "If true, use the reduced PM value set in KPM indications",
                          BooleanValue(false),
                          MakeBooleanAccessor(&NrGnbNetDevice::m_reducedPmValues),
                          MakeBooleanChecker());
    return tid;
}

NrGnbNetDevice::NrGnbNetDevice()
    : m_forceE2FileLogging(false),
      m_cellId(0),
      m_e2Periodicity(1.0),
      m_sendCuUp(true),
      m_sendCuCp(true),
      m_sendDu(false),
      m_reducedPmValues(false),
      m_stopSendingMessages(false),
      m_isReportingEnabled(false)
{
    NS_LOG_FUNCTION(this);
}

void
NrGnbNetDevice::stopSendingAndCancelSchedule()
{
    m_stopSendingMessages = true;
}

void
NrGnbNetDevice::KpmSubscriptionCallback(E2AP_PDU_t* sub_req_pdu)
{
    NS_LOG_DEBUG("Received RIC Subscription Request, cellId= " << m_cellId);

    m_lastSubscriptionParams = m_e2term->ProcessRicSubscriptionRequest(sub_req_pdu);
    m_hasValidSubscription = true;
    NS_LOG_DEBUG("requestorId " << +m_lastSubscriptionParams.requestorId << ", instanceId "
                                << +m_lastSubscriptionParams.instanceId << ", ranFuncionId "
                                << +m_lastSubscriptionParams.ranFuncionId << ", actionId "
                                << +m_lastSubscriptionParams.actionId);

    if (!m_stopSendingMessages && !m_isReportingEnabled && !m_forceE2FileLogging)
    {
        m_isReportingEnabled = true;
        // This callback runs on the e2sim SCTP thread; hand the periodic
        // report generation over to the simulator thread.
        Simulator::ScheduleWithContext(1,
                                       MilliSeconds(0),
                                       &NrGnbNetDevice::BuildAndSendReportMessage,
                                       this,
                                       m_lastSubscriptionParams);
    }
}

void
NrGnbNetDevice::ControlMessageReceivedCallback(E2AP_PDU_t* sub_req_pdu)
{
    NS_LOG_DEBUG("NrGnbNetDevice::ControlMessageReceivedCallback: Received RIC Control Message");

    Ptr<RicControlMessage> controlMessage = Create<RicControlMessage>(sub_req_pdu);
    NS_LOG_INFO("Request type " << controlMessage->m_requestType);

    switch (controlMessage->m_e2SmRcControlHeaderFormat1->ric_Style_Type)
    {
    case RicControlMessage::ControlMessageServiceStyle::Connected_Mode_Mobility: {
        if (controlMessage->m_e2SmRcControlHeaderFormat1->ric_ControlAction_ID !=
            RicControlMessage::Connected_Mode_Mobility_Control_Action_ID::Handover_Control)
        {
            NS_LOG_UNCOND("Unsupported Connected_Mode_Mobility action "
                          << controlMessage->m_e2SmRcControlHeaderFormat1->ric_ControlAction_ID);
            break;
        }

        UEID_GNB_t* ueGnb = controlMessage->m_e2SmRcControlHeaderFormat1->ueID.choice.gNB_UEID;
        uint64_t imsi = 0;
        memcpy(&imsi, ueGnb->ran_UEID->buf, ueGnb->ran_UEID->size);
        uint16_t targetCellId = controlMessage->GetTargetCell();

        // find the RNTI of the UE with this IMSI in our cell
        uint16_t rnti = 0;
        bool ueFound = false;
        for (const auto& ue : m_rrc->GetUeMap())
        {
            if (ue.second->GetImsi() == imsi)
            {
                rnti = ue.first;
                ueFound = true;
                break;
            }
        }
        if (!ueFound)
        {
            NS_LOG_UNCOND("[E2 HO] UE with IMSI " << imsi << " not found in cell " << m_cellId
                                                  << "; control ignored");
            break;
        }

        NS_LOG_UNCOND("[E2 HO] cell " << m_cellId << ": handover UE imsi=" << imsi
                                      << " rnti=" << rnti << " -> cell " << targetCellId);
        // The E2 callback runs on the e2sim SCTP thread; hand the RRC action
        // over to the simulator thread. Uses nr's native X2 handover trigger.
        Simulator::ScheduleWithContext(1,
                                       Seconds(0),
                                       &NrGnbRrc::SendHandoverRequest,
                                       m_rrc,
                                       rnti,
                                       targetCellId);
        break;
    }
    case RicControlMessage::ControlMessageServiceStyle::Antenna_Control: {
        if (controlMessage->m_e2SmRcControlHeaderFormat1->ric_ControlAction_ID !=
            RicControlMessage::Antenna_Control_Action_ID::RET_Tilt_Control)
        {
            NS_LOG_UNCOND("Unsupported Antenna Control action "
                          << controlMessage->m_e2SmRcControlHeaderFormat1->ric_ControlAction_ID);
            break;
        }
        RicControlMessage::RetControl ret = controlMessage->GetRetControl();
        if (!ret.valid)
        {
            NS_LOG_UNCOND("[RET] Malformed RET control message; ignored");
            break;
        }
        if (ret.cellId != m_cellId)
        {
            NS_LOG_UNCOND("[RET] control for cell " << ret.cellId << " received by cell "
                                                    << m_cellId << "; ignored");
            break;
        }
        Simulator::ScheduleWithContext(1,
                                       Seconds(0),
                                       &NrGnbNetDevice::ApplyRetControl,
                                       this,
                                       ret.tiltDeg,
                                       ret.hasBearing,
                                       ret.bearingDeg);
        break;
    }
    case RicControlMessage::ControlMessageServiceStyle::Energy_state: {
        long actionId = controlMessage->m_e2SmRcControlHeaderFormat1->ric_ControlAction_ID;
        EnergyState state = EnergyState::ON;
        bool recognized = true;
        switch (actionId)
        {
        case RicControlMessage::Energy_State_Control_Action_ID::Cell_Off:
            state = EnergyState::OFF;
            break;
        case RicControlMessage::Energy_State_Control_Action_ID::Cell_On:
            state = EnergyState::ON;
            break;
        case RicControlMessage::Energy_State_Control_Action_ID::Cell_Sleep:
            state = EnergyState::SLEEP;
            break;
        default:
            recognized = false;
            break;
        }
        if (!recognized)
        {
            NS_LOG_UNCOND("Unsupported Energy_state action " << actionId);
            break;
        }

        uint16_t targetCellId = controlMessage->GetTargetCell();
        if (targetCellId != m_cellId)
        {
            NS_LOG_UNCOND("[ES] energy-state control for cell "
                          << targetCellId << " received by cell " << m_cellId << "; ignored");
            break;
        }

        Simulator::ScheduleWithContext(1,
                                       Seconds(0),
                                       &NrGnbNetDevice::ApplyEnergyState,
                                       this,
                                       state);
        break;
    }
    default: {
        NS_LOG_UNCOND("Unsupported RIC Style Type "
                      << controlMessage->m_e2SmRcControlHeaderFormat1->ric_Style_Type);
        break;
    }
    }
}

void
NrGnbNetDevice::ApplyRetControl(double tiltDeg, bool hasBearing, double bearingDeg)
{
    // With the official Sionna RT channel model, the UniformPlanarArray
    // orientation (BearingAngle/DowntiltAngle) is passed to the ray tracer as
    // the Sionna Transmitter orientation on every channel update
    // (SionnaRtChannelModel::CreateScene), so this change has a real
    // propagation effect from the next update period onwards.
    uint32_t applied = 0;
    for (uint32_t i = 0; i < GetCcMapSize(); ++i)
    {
        Ptr<NrGnbPhy> phy = GetPhy(i);
        if (!phy || !phy->GetSpectrumPhy())
        {
            continue;
        }
        Ptr<UniformPlanarArray> antenna =
            DynamicCast<UniformPlanarArray>(phy->GetSpectrumPhy()->GetAntenna());
        if (!antenna)
        {
            continue;
        }
        antenna->SetAttribute("DowntiltAngle", DoubleValue(tiltDeg * M_PI / 180.0));
        if (hasBearing)
        {
            antenna->SetAttribute("BearingAngle", DoubleValue(bearingDeg * M_PI / 180.0));
        }
        ++applied;
    }
    NS_LOG_UNCOND("[RET] cell " << m_cellId << ": applied tilt=" << tiltDeg << " deg"
                                << (hasBearing
                                        ? " bearing=" + std::to_string(bearingDeg) + " deg"
                                        : "")
                                << " to " << applied << " BWP antenna(s)");
}

void
NrGnbNetDevice::ApplyEnergyState(EnergyState state)
{
    // -100 dBm is far below any receiver's noise floor, so this silences the
    // cell in both the standard propagation-loss and Sionna RT channel
    // paths (both consume NrGnbPhy::m_txPower identically) without touching
    // MAC/PHY scheduling logic at all. SLEEP uses a smaller, fixed
    // attenuation from the nominal power as a simple middle state.
    constexpr double kOffTxPowerDbm = -100.0;
    constexpr double kSleepAttenuationDb = 20.0;

    m_energyState = state;
    uint32_t applied = 0;
    for (uint32_t i = 0; i < GetCcMapSize(); ++i)
    {
        Ptr<NrGnbPhy> phy = GetPhy(i);
        if (!phy)
        {
            continue;
        }

        if (state == EnergyState::ON)
        {
            if (m_txPowerSaved)
            {
                phy->SetTxPower(m_savedTxPowerDbm);
                m_txPowerSaved = false;
            }
        }
        else
        {
            if (!m_txPowerSaved)
            {
                m_savedTxPowerDbm = phy->GetTxPower();
                m_txPowerSaved = true;
            }
            phy->SetTxPower(state == EnergyState::OFF ? kOffTxPowerDbm
                                                       : m_savedTxPowerDbm - kSleepAttenuationDb);
        }
        ++applied;
    }

    const char* stateName =
        state == EnergyState::ON ? "ON" : (state == EnergyState::OFF ? "OFF" : "SLEEP");
    NS_LOG_UNCOND("[ES] cell " << m_cellId << ": energy state -> " << stateName << " applied to "
                               << applied << " BWP PHY(s)");
}

void
NrGnbNetDevice::SetE2Termination(Ptr<E2Termination> e2term)
{
    m_e2term = e2term;

    NS_LOG_DEBUG("Register E2SM NR");

    if (!m_forceE2FileLogging)
    {
        long m_e2_func_id = static_cast<long>(e2_func_id);
        long m_rc_e2_func_id = static_cast<long>(rc_e2_func_id);
        Ptr<KpmFunctionDescription> kpmFd = Create<KpmFunctionDescription>();
        e2term->RegisterKpmCallbackToE2Sm(
            m_e2_func_id,
            kpmFd,
            std::bind(&NrGnbNetDevice::KpmSubscriptionCallback, this, std::placeholders::_1));

        Ptr<RicControlFunctionDescription> ricCtrlFd = Create<RicControlFunctionDescription>();
        e2term->RegisterSmCallbackToE2Sm(
            m_rc_e2_func_id,
            ricCtrlFd,
            std::bind(&NrGnbNetDevice::ControlMessageReceivedCallback, this, std::placeholders::_1));

        e2term->RegisterCallbackFunctionToE2Sm(
            1,
            std::bind(&NrGnbNetDevice::stopSendingAndCancelSchedule, this));
    }
}

Ptr<E2Termination>
NrGnbNetDevice::GetE2Termination() const
{
    return m_e2term;
}

std::string
NrGnbNetDevice::GetImsiString(uint64_t imsi)
{
    std::string ueImsi = std::to_string(imsi);
    std::string ueImsiComplete{};
    if (ueImsi.length() == 1)
    {
        ueImsiComplete = "0000" + ueImsi;
    }
    else if (ueImsi.length() == 2)
    {
        ueImsiComplete = "000" + ueImsi;
    }
    else
    {
        ueImsiComplete = "00" + ueImsi;
    }
    return ueImsiComplete;
}

Ptr<KpmIndicationHeader>
NrGnbNetDevice::BuildRicIndicationHeader(std::string plmId, std::string gnbId, uint16_t nrCellId)
{
    if (m_forceE2FileLogging)
    {
        return nullptr;
    }
    KpmIndicationHeader::KpmRicIndicationHeaderValues headerValues;
    headerValues.m_plmId = plmId;
    headerValues.m_gnbId = gnbId;
    headerValues.m_nrCellId = nrCellId;
    const uint64_t timestamp = m_startTime + (uint64_t)Simulator::Now().GetMilliSeconds();
    NS_LOG_DEBUG("NR plmid " << plmId << " gnbId " << gnbId << " nrCellId " << nrCellId
                             << " timestamp " << timestamp);
    headerValues.m_timestamp = timestamp;
    return Create<KpmIndicationHeader>(KpmIndicationHeader::GlobalE2nodeType::gNB, headerValues);
}

Ptr<KpmIndicationMessage>
NrGnbNetDevice::BuildRicIndicationMessageCuUp(std::string plmId)
{
    Ptr<MmWaveIndicationMessageHelper> indicationMessageHelper =
        CreateObject<MmWaveIndicationMessageHelper>(IndicationMessageHelper::IndicationMessageType::CuUp,
                                              m_forceE2FileLogging,
                                              m_reducedPmValues);

    auto ueMap = m_rrc->GetUeMap();
    double cellDlTxVolume = 0;

    for (const auto& ue : ueMap)
    {
        uint64_t imsi = ue.second->GetImsi();
        std::string ueImsiComplete = GetImsiString(imsi);

        // Aggregate DL PDCP and RLC statistics over this UE's actual data
        // bearers. Unlike the LTE-based ns3-o-ran-e2 fork (which hardcoded
        // LCID 3), nr maps LCID = DRB id, so read each bearer's LCID.
        long txDlPackets = 0;
        double txBytes = 0;
        long txPdcpPduNrRlc = 0;
        double txPdcpPduBytesNrRlc = 0;

        for (const auto& drb : ue.second->GetDrbMap())
        {
            const uint8_t lcid = drb.second->m_logicalChannelIdentity;
            if (m_e2PdcpStatsCalculator)
            {
                txDlPackets += m_e2PdcpStatsCalculator->GetDlTxPackets(imsi, lcid);
                txBytes += m_e2PdcpStatsCalculator->GetDlTxData(imsi, lcid) * 8 / 1e3; // kbit
                m_e2PdcpStatsCalculator->ResetResultsForImsiLcid(imsi, lcid);
            }
            if (m_e2RlcStatsCalculator)
            {
                txPdcpPduNrRlc += m_e2RlcStatsCalculator->GetDlTxPackets(imsi, lcid);
                txPdcpPduBytesNrRlc +=
                    m_e2RlcStatsCalculator->GetDlTxData(imsi, lcid) * 8 / 1e3; // kbit
                m_e2RlcStatsCalculator->ResetResultsForImsiLcid(imsi, lcid);
            }
        }
        cellDlTxVolume += txBytes;

        NS_LOG_DEBUG(Simulator::Now().GetSeconds()
                     << " " << m_cellId << " cell, connected UE with IMSI " << imsi
                     << " txDlPackets " << txDlPackets << " txBytes " << txBytes
                     << " txDlPacketsNr " << txPdcpPduNrRlc << " txDlBytesNr "
                     << txPdcpPduBytesNrRlc);

        indicationMessageHelper->AddCuUpUePmItem(ueImsiComplete,
                                                 txPdcpPduBytesNrRlc,
                                                 txPdcpPduNrRlc);
    }

    indicationMessageHelper->FillCuUpValues(plmId);

    NS_LOG_DEBUG(Simulator::Now().GetSeconds()
                 << " " << m_cellId << " cell DL tx volume " << cellDlTxVolume << " kbit");

    return indicationMessageHelper->CreateIndicationMessage(m_e2term->SubscriptionMapRef());
}

Ptr<KpmIndicationMessage>
NrGnbNetDevice::BuildRicIndicationMessageCuCp(std::string plmId)
{
    Ptr<MmWaveIndicationMessageHelper> indicationMessageHelper =
        CreateObject<MmWaveIndicationMessageHelper>(IndicationMessageHelper::IndicationMessageType::CuCp,
                                              m_forceE2FileLogging,
                                              m_reducedPmValues);

    auto ueMap = m_rrc->GetUeMap();

    for (const auto& ue : ueMap)
    {
        uint64_t imsi = ue.second->GetImsi();
        std::string ueImsiComplete = GetImsiString(imsi);
        long numDrb = ue.second->GetDrbMap().size();

        // Serving cell: use the last RSRP reported for our own cellId, if
        // this UE's RRC measurements have been received yet (requires an
        // RSRP-based handover algorithm to be installed -- see
        // RecvMeasurementReport). Otherwise report 0, matching the previous
        // stub behavior, rather than fabricating a value.
        double servingRsrpDbm = 0;
        auto imsiIt = m_l3RsrpDbmMap.find(imsi);
        bool haveMeasurements = imsiIt != m_l3RsrpDbmMap.end();
        if (haveMeasurements)
        {
            auto servingIt = imsiIt->second.find(m_cellId);
            if (servingIt != imsiIt->second.end())
            {
                servingRsrpDbm = servingIt->second;
            }
        }
        Ptr<L3RrcMeasurements> l3RrcMeasurementServing =
            L3RrcMeasurements::CreateL3RrcUeSpecificSinrServing(m_cellId,
                                                                m_cellId,
                                                                servingRsrpDbm);

        // Neighbours: every other cellId this UE has reported RSRP for,
        // strongest first, capped at the standard's max of 8 reported cells
        // (E2SM-KPM UE Measurement Report, matching the old
        // MmWaveEnbNetDevice::E2SM_REPORT_MAX_NEIGH constant).
        constexpr uint16_t kMaxNeighbours = 8;
        Ptr<L3RrcMeasurements> l3RrcMeasurementNeigh =
            L3RrcMeasurements::CreateL3RrcUeSpecificSinrNeigh();
        if (haveMeasurements)
        {
            std::multimap<double, uint16_t, std::greater<>> byStrength;
            for (const auto& cellRsrp : imsiIt->second)
            {
                if (cellRsrp.first != m_cellId)
                {
                    byStrength.emplace(cellRsrp.second, cellRsrp.first);
                }
            }
            uint16_t added = 0;
            for (const auto& entry : byStrength)
            {
                if (added >= kMaxNeighbours)
                {
                    break;
                }
                l3RrcMeasurementNeigh->AddNeighbourCellMeasurement(entry.second, entry.first);
                ++added;
            }
        }

        indicationMessageHelper->AddCuCpUePmItem(ueImsiComplete,
                                                 numDrb,
                                                 0,
                                                 l3RrcMeasurementServing,
                                                 l3RrcMeasurementNeigh);
    }

    indicationMessageHelper->FillCuCpValues(ueMap.size());

    return indicationMessageHelper->CreateIndicationMessage(m_e2term->SubscriptionMapRef());
}

Ptr<KpmIndicationMessage>
NrGnbNetDevice::BuildRicIndicationMessageDu(std::string plmId)
{
    Ptr<MmWaveIndicationMessageHelper> indicationMessageHelper =
        CreateObject<MmWaveIndicationMessageHelper>(IndicationMessageHelper::IndicationMessageType::Du,
                                              m_forceE2FileLogging,
                                              m_reducedPmValues);

    // Same ratio scratch/khu-ret-experiment.cc's ReportKpiToInflux already
    // computes for its cell_kpi dashboard's prb_utilization_pct field.
    double prbUtilizationDl = (m_duPrbCapacityAccum > 0)
                                  ? (100.0 * static_cast<double>(m_duPrbUsedAccum) /
                                     static_cast<double>(m_duPrbCapacityAccum))
                                  : 0.0;

    auto ueMap = m_rrc->GetUeMap();

    // Physical PRB count for this cell's (single) BWP -- static per-numerology/
    // bandwidth capacity, not derived from the accumulators above.
    long totalPrbDl = GetPhy(0) ? static_cast<long>(GetBwpDlBandwidth(0)) : 0;

    // Fields with no cheap real data source yet (per-cell PDU/QAM/retx
    // counts, SINR bins, RLC buffer occupancy) are reported as 0; only
    // PRB utilization, MCS distribution and active-UE count are real.
    indicationMessageHelper->AddDuCellPmItem(0,                    // macPduCellSpecific
                                             0,                    // macPduInitialCellSpecific
                                             0,                    // macQpskCellSpecific
                                             0,                    // mac16QamCellSpecific
                                             0,                    // mac64QamCellSpecific
                                             prbUtilizationDl,
                                             totalPrbDl,
                                             0,                    // macRetxCellSpecific
                                             0,                    // macVolumeCellSpecific
                                             m_duMcsBins[0],
                                             m_duMcsBins[1],
                                             m_duMcsBins[2],
                                             m_duMcsBins[3],
                                             m_duMcsBins[4],
                                             m_duMcsBins[5],
                                             0,                    // macSinrBin1CellSpecific
                                             0,                    // macSinrBin2CellSpecific
                                             0,                    // macSinrBin3CellSpecific
                                             0,                    // macSinrBin4CellSpecific
                                             0,                    // macSinrBin5CellSpecific
                                             0,                    // macSinrBin6CellSpecific
                                             0,                    // macSinrBin7CellSpecific
                                             0,                    // rlcBufferOccupCellSpecific
                                             static_cast<long>(ueMap.size()));

    indicationMessageHelper->FillDuValues(std::to_string(m_cellId));

    // Reset accumulators for the next reporting period.
    m_duPrbUsedAccum = 0;
    m_duPrbCapacityAccum = 0;
    m_duMcsBins.fill(0);

    return indicationMessageHelper->CreateIndicationMessage(m_e2term->SubscriptionMapRef());
}

void
NrGnbNetDevice::BuildAndSendReportMessage(E2Termination::RicSubscriptionRequest_rval_s params)
{
    if (m_stopSendingMessages)
    {
        return;
    }

    std::string plmId = "111";
    std::string gnbId = std::to_string(m_cellId);

    NS_LOG_DEBUG("NrGnbNetDevice " << m_cellId << " BuildAndSendReportMessage at time "
                                   << Simulator::Now().GetSeconds());

    if (m_sendCuUp)
    {
        Ptr<KpmIndicationHeader> header = BuildRicIndicationHeader(plmId, gnbId, m_cellId);
        Ptr<KpmIndicationMessage> cuUpMsg = BuildRicIndicationMessageCuUp(plmId);
        if (header && cuUpMsg)
        {
            NS_LOG_DEBUG("Send NR CU-UP");
            auto* pdu = new E2AP_PDU(); // value-init: ASN.1 C struct must start zeroed
            encoding::generate_e2apv1_indication_request_parameterized(
                pdu,
                params.requestorId,
                params.instanceId,
                params.ranFuncionId,
                params.actionId,
                1, // sequence number
                (uint8_t*)header->m_buffer,
                header->m_size,
                (uint8_t*)cuUpMsg->m_buffer,
                cuUpMsg->m_size);
            m_e2term->SendE2Message(pdu);
            delete pdu;
        }
    }

    if (m_sendCuCp)
    {
        Ptr<KpmIndicationHeader> header = BuildRicIndicationHeader(plmId, gnbId, m_cellId);
        Ptr<KpmIndicationMessage> cuCpMsg = BuildRicIndicationMessageCuCp(plmId);
        if (header && cuCpMsg)
        {
            NS_LOG_DEBUG("Send NR CU-CP");
            auto* pdu = new E2AP_PDU(); // value-init: ASN.1 C struct must start zeroed
            encoding::generate_e2apv1_indication_request_parameterized(
                pdu,
                params.requestorId,
                params.instanceId,
                params.ranFuncionId,
                params.actionId,
                1, // sequence number
                (uint8_t*)header->m_buffer,
                header->m_size,
                (uint8_t*)cuCpMsg->m_buffer,
                cuCpMsg->m_size);
            m_e2term->SendE2Message(pdu);
            delete pdu;
        }
    }

    if (m_sendDu)
    {
        Ptr<KpmIndicationHeader> header = BuildRicIndicationHeader(plmId, gnbId, m_cellId);
        Ptr<KpmIndicationMessage> duMsg = BuildRicIndicationMessageDu(plmId);
        if (header && duMsg)
        {
            NS_LOG_DEBUG("Send NR DU");
            auto* pdu = new E2AP_PDU(); // value-init: ASN.1 C struct must start zeroed
            encoding::generate_e2apv1_indication_request_parameterized(
                pdu,
                params.requestorId,
                params.instanceId,
                params.ranFuncionId,
                params.actionId,
                1, // sequence number
                (uint8_t*)header->m_buffer,
                header->m_size,
                (uint8_t*)duMsg->m_buffer,
                duMsg->m_size);
            m_e2term->SendE2Message(pdu);
            delete pdu;
        }
    }

    if (!m_stopSendingMessages && m_hasValidSubscription)
    {
        Simulator::Schedule(Seconds(m_e2Periodicity),
                            &NrGnbNetDevice::BuildAndSendReportMessage,
                            this,
                            params);
    }
}

NrGnbNetDevice::~NrGnbNetDevice()
{
    NS_LOG_FUNCTION(this);
}

Ptr<NrMacScheduler>
NrGnbNetDevice::GetScheduler(uint8_t index) const
{
    NS_LOG_FUNCTION(this);
    return m_ccMap.at(index)->GetScheduler();
}

void
NrGnbNetDevice::SetCcMap(const std::map<uint8_t, Ptr<BandwidthPartGnb>>& ccm)
{
    NS_ABORT_IF(!m_ccMap.empty());
    m_ccMap = ccm;
}

uint32_t
NrGnbNetDevice::GetCcMapSize() const
{
    return static_cast<uint32_t>(m_ccMap.size());
}

void
NrGnbNetDevice::SetNrFhControl(Ptr<NrFhControl> nrFh)
{
    NS_LOG_FUNCTION(this);
    m_nrFhControl = nrFh;
}

Ptr<NrFhControl>
NrGnbNetDevice::GetNrFhControl()
{
    NS_LOG_FUNCTION(this);
    return m_nrFhControl;
}

void
NrGnbNetDevice::RouteIngoingCtrlMsgs(const std::list<Ptr<NrControlMessage>>& msgList,
                                     uint32_t sourceBwpArfcn)
{
    NS_LOG_FUNCTION(this);

    for (const auto& msg : msgList)
    {
        uint32_t bwpArfcn = DynamicCast<BwpManagerGnb>(m_componentCarrierManager)
                                ->RouteIngoingCtrlMsgs(msg, sourceBwpArfcn);
        auto bwpIt = std::find_if(m_ccMap.begin(), m_ccMap.end(), [bwpArfcn](const auto& bwp) {
            return bwp.second->GetArfcn() == bwpArfcn;
        });
        NS_ASSERT_MSG(bwpIt != m_ccMap.end(), "gNB missing BWP with ARFCN: " << bwpArfcn);
        bwpIt->second->GetPhy()->PhyCtrlMessagesReceived(msg);
    }
}

void
NrGnbNetDevice::RouteOutgoingCtrlMsgs(const std::list<Ptr<NrControlMessage>>& msgList,
                                      uint8_t sourceBwpId)
{
    NS_LOG_FUNCTION(this);

    for (const auto& msg : msgList)
    {
        uint8_t bwpId = DynamicCast<BwpManagerGnb>(m_componentCarrierManager)
                            ->RouteOutgoingCtrlMsg(msg, sourceBwpId);
        NS_ASSERT_MSG(m_ccMap.size() > bwpId,
                      "Returned bwp " << +bwpId << " is not present. Check your configuration");
        NS_ASSERT_MSG(
            m_ccMap.at(bwpId)->GetPhy()->HasDlSlot(),
            "Returned bwp "
                << +bwpId
                << " has no DL slot, so the message can't go out. Check your configuration");
        m_ccMap.at(bwpId)->GetPhy()->EncodeCtrlMsg(msg);
    }
}

void
NrGnbNetDevice::DoInitialize()
{
    NS_LOG_FUNCTION(this);
    if (!m_isCellConfigured)
    {
        ConfigureCell();
    }

    if (m_sendCuCp)
    {
        // Config::Connect matches by device TypeId + object path, so this
        // fires once per NrGnbNetDevice::DoInitialize() call and attaches
        // exactly this device's callback to exactly this device's RRC.
        Config::ConnectWithoutContextFailSafe(
            "/NodeList/*/DeviceList/*/NrGnbRrc/RecvMeasurementReport",
            MakeCallback(&NrGnbNetDevice::RecvMeasurementReport, this));
    }

    if (m_sendDu)
    {
        // NrGnbPhy's SlotDataStats gives the same usedReg/availableRb*
        // availableSym quantities scratch/khu-ret-experiment.cc already
        // uses for its cell_kpi PRB utilization dashboard -- reuse that
        // exact, already-proven Config path.
        Config::ConnectWithoutContextFailSafe(
            "/NodeList/*/DeviceList/*/BandwidthPartMap/*/NrGnbPhy/SlotDataStats",
            MakeCallback(&NrGnbNetDevice::NotifySlotDataStats, this));

        // NrGnbMac objects live inside the per-CC BandwidthPartGnb map
        // rather than as a directly addressable Config path the way PHY is
        // above, so connect directly through the already-available GetMac()
        // accessor for the MCS histogram.
        for (uint32_t i = 0; i < GetCcMapSize(); ++i)
        {
            Ptr<NrGnbMac> mac = GetMac(i);
            if (mac)
            {
                mac->TraceConnectWithoutContext("DlScheduling",
                                                MakeCallback(&NrGnbNetDevice::NotifyDlScheduling,
                                                            this));
            }
        }
    }
}

void
NrGnbNetDevice::RecvMeasurementReport(uint64_t imsi,
                                      uint16_t cellId,
                                      uint16_t rnti,
                                      NrRrcSap::MeasurementReport report)
{
    // Only record reports that reached this device's own cell; the
    // Config::Connect wildcard above matches every gNB's RRC, so without
    // this check every NrGnbNetDevice would collect every other cell's
    // reports too.
    if (cellId != m_cellId)
    {
        return;
    }

    // RSRP is 3GPP-encoded as 0..97 mapping linearly to -140..-44 dBm
    // (TS 36.133 9.1.4). There is no per-neighbour SINR in the standard RRC
    // measurement report the way the legacy mmwave+LteEnbRrc dual-connectivity
    // trace exposed one directly, so RSRP is what feeds the KPM CU-CP
    // "RS-SINR" fields here.
    const auto& pCell = report.measResults.measResultPCell;
    m_l3RsrpDbmMap[imsi][cellId] = static_cast<double>(pCell.rsrpResult) - 140.0;

    if (report.measResults.haveMeasResultNeighCells)
    {
        for (const auto& neigh : report.measResults.measResultListEutra)
        {
            if (neigh.haveRsrpResult)
            {
                m_l3RsrpDbmMap[imsi][neigh.physCellId] =
                    static_cast<double>(neigh.rsrpResult) - 140.0;
            }
        }
    }

    NS_LOG_DEBUG("cell " << m_cellId << " UE imsi=" << imsi << " rnti=" << rnti
                         << " serving RSRP " << m_l3RsrpDbmMap[imsi][cellId] << " dBm, "
                         << report.measResults.measResultListEutra.size()
                         << " neighbour(s) reported");
}

void
NrGnbNetDevice::NotifyDlScheduling(NrSchedulingCallbackInfo info)
{
    if (info.m_mcs != UINT8_MAX)
    {
        std::size_t bin = std::min<std::size_t>(5, info.m_mcs / 5);
        ++m_duMcsBins[bin];
    }
}

void
NrGnbNetDevice::NotifySlotDataStats(const SfnSf& /* sfnSf */,
                                    uint32_t /* scheduledUe */,
                                    uint32_t usedReg,
                                    uint32_t /* usedSym */,
                                    uint32_t availableRb,
                                    uint32_t availableSym,
                                    uint16_t /* bwpId */,
                                    uint16_t cellId)
{
    // The Config::Connect wildcard in DoInitialize matches every gNB's PHY,
    // same reasoning as RecvMeasurementReport's cellId filter above.
    if (cellId != m_cellId)
    {
        return;
    }
    m_duPrbUsedAccum += usedReg;
    m_duPrbCapacityAccum += static_cast<uint64_t>(availableRb) * availableSym;
}

void
NrGnbNetDevice::DoDispose()
{
    NS_LOG_FUNCTION(this);

    m_rrc->Dispose();
    m_rrc = nullptr;
    for (const auto& it : m_ccMap)
    {
        it.second->Dispose();
    }
    m_ccMap.clear();
    m_componentCarrierManager->Dispose();
    m_componentCarrierManager = nullptr;
    NrNetDevice::DoDispose();
}

Ptr<NrGnbMac>
NrGnbNetDevice::GetMac(uint8_t index) const
{
    return m_ccMap.at(index)->GetMac();
}

Ptr<NrGnbPhy>
NrGnbNetDevice::GetPhy(uint8_t index) const
{
    NS_LOG_FUNCTION(this);
    return m_ccMap.at(index)->GetPhy();
}

Ptr<BwpManagerGnb>
NrGnbNetDevice::GetBwpManager() const
{
    return DynamicCast<BwpManagerGnb>(m_componentCarrierManager);
}

uint16_t
NrGnbNetDevice::GetCellId() const
{
    NS_LOG_FUNCTION(this);
    return m_cellId;
}

std::vector<uint16_t>
NrGnbNetDevice::GetBwpIds() const
{
    std::vector<uint16_t> bwpIds;

    bwpIds.reserve(m_ccMap.size());
    for (auto& it : m_ccMap)
    {
        bwpIds.push_back(it.second->GetBwpId());
    }
    return bwpIds;
}

void
NrGnbNetDevice::SetCellId(uint16_t cellId)
{
    NS_LOG_FUNCTION(this);
    m_cellId = cellId;
}

uint16_t
NrGnbNetDevice::GetCellId(uint8_t index) const
{
    NS_LOG_FUNCTION(this);
    return m_ccMap.at(index)->GetCellId();
}

void
NrGnbNetDevice::SetRrc(Ptr<NrGnbRrc> rrc)
{
    m_rrc = rrc;
}

Ptr<NrGnbRrc>
NrGnbNetDevice::GetRrc()
{
    return m_rrc;
}

bool
NrGnbNetDevice::DoSend(Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{
    NS_LOG_FUNCTION(this << packet << dest << protocolNumber);
    NS_ABORT_MSG_IF(protocolNumber != Ipv4L3Protocol::PROT_NUMBER &&
                        protocolNumber != Ipv6L3Protocol::PROT_NUMBER,
                    "unsupported protocol " << protocolNumber
                                            << ", only IPv4 and IPv6 are supported");

    NS_LOG_INFO("Forward received packet to RRC Layer");
    m_txTrace(packet, dest);

    return m_rrc->SendData(packet);
}

void
NrGnbNetDevice::UpdateConfig()
{
    NS_LOG_FUNCTION(this);
    // No longer does anything; replaced by ConfigureCell()
}

void
NrGnbNetDevice::ConfigureCell()
{
    NS_LOG_FUNCTION(this);
    NS_ASSERT_MSG(!m_isCellConfigured, "ConfigureCell() has already been called");
    NS_ASSERT_MSG(!m_ccMap.empty(), "Component carrier map is empty");
    m_isCellConfigured = true;
    m_rrc->ConfigureCell(m_ccMap);
    m_handoverAlgorithm->Initialize();

    if (m_sendCuCp)
    {
        // KPM CU-CP indications need L3 RSRP measurements independent of
        // whether a handover algorithm is installed (the default
        // NrNoOpHandoverAlgorithm configures none at all, and even an
        // RSRP-based one only reports on its own event trigger, not on a
        // predictable schedule). nr's UE-side RRC only implements
        // EVENT-triggered reporting (NrUeRrc::ApplyMeasConfig asserts on
        // PERIODICAL), so approximate "periodic" with EVENT_A4 against the
        // lowest possible RSRP threshold -- any detected cell satisfies it
        // immediately, and 3GPP event-triggered reporting still repeats
        // every reportInterval for reportAmount occurrences once triggered,
        // giving effectively periodic reports without interfering with any
        // handover-triggering config the handover algorithm may have added
        // above. EVENT_A4 (not A1) is required for neighbor-cell reporting:
        // A1 only ever triggers on the serving cell itself (cellsTriggeredList
        // contains just m_cellId), which NrUeRrc's report builder then filters
        // out entirely when it looks for non-serving entries, leaving
        // measResultListEutra empty. A4 iterates all non-serving stored
        // measurements and triggers each one independently on an absolute
        // threshold, so neighbor cells actually show up in the report.
        NrRrcSap::ReportConfigEutra reportConfig;
        reportConfig.triggerType = NrRrcSap::ReportConfigEutra::EVENT;
        reportConfig.eventId = NrRrcSap::ReportConfigEutra::EVENT_A4;
        reportConfig.threshold1.choice = NrRrcSap::ThresholdEutra::THRESHOLD_RSRP;
        reportConfig.threshold1.range = 0;
        reportConfig.timeToTrigger = 0;
        reportConfig.triggerQuantity = NrRrcSap::ReportConfigEutra::RSRP;
        reportConfig.reportQuantity = NrRrcSap::ReportConfigEutra::BOTH;
        reportConfig.maxReportCells = 8;
        reportConfig.reportInterval = NrRrcSap::ReportConfigEutra::MS1024;
        reportConfig.reportAmount = 0xff;
        m_rrc->AddUeMeasReportConfig(reportConfig);
    }

    if (m_e2term)
    {
        NS_LOG_DEBUG("E2sim start in cell " << m_cellId << " force CSV logging "
                                            << m_forceE2FileLogging);
        if (!m_forceE2FileLogging)
        {
            Simulator::Schedule(MicroSeconds(0), &E2Termination::Start, m_e2term);
        }
    }
}

bool
NrGnbNetDevice::IsCellConfigured() const
{
    return m_isCellConfigured;
}

uint16_t
NrGnbNetDevice::GetBwpDlBandwidth(uint16_t bwpId) const
{
    if (m_rrc->HasBwpId(bwpId))
    {
        for (const auto& [key, cc] : m_ccMap)
        {
            if (cc->GetBwpId() == bwpId)
            {
                return cc->GetDlBandwidth();
            }
        }
    }
    return 0;
}

uint16_t
NrGnbNetDevice::GetBwpUlBandwidth(uint16_t bwpId) const
{
    NS_ASSERT_MSG(m_rrc->HasBwpId(bwpId), "Unknown bwpId");
    if (m_rrc->HasBwpId(bwpId))
    {
        for (const auto& [key, cc] : m_ccMap)
        {
            if (cc->GetBwpId() == bwpId)
            {
                return cc->GetUlBandwidth();
            }
        }
    }
    return 0;
}

uint32_t
NrGnbNetDevice::GetBwpArfcn(uint16_t bwpId) const
{
    NS_ASSERT_MSG(m_rrc->HasBwpId(bwpId), "Unknown bwpId");
    if (m_rrc->HasBwpId(bwpId))
    {
        for (const auto& [key, cc] : m_ccMap)
        {
            if (cc->GetBwpId() == bwpId)
            {
                return cc->GetArfcn();
            }
        }
    }
    return 0;
}

uint16_t
NrGnbNetDevice::GetArfcnBwpId(uint32_t arfcn) const
{
    for (std::size_t i = 0; i < m_ccMap.size(); i++)
    {
        if (m_ccMap.at(i)->GetArfcn() == arfcn)
        {
            return (uint16_t)i;
        }
    }
    NS_ABORT_MSG("gNB should have the searched arfcn");
}

} // namespace ns3
