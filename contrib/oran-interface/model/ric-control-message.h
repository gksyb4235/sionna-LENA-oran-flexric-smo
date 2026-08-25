/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2022 Northeastern University
 * Copyright (c) 2022 Sapienza, University of Rome
 * Copyright (c) 2022 University of Padova
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Andrea Lacava <thecave003@gmail.com>
 *		   Tommaso Zugno <tommasozugno@gmail.com>
 *		   Michele Polese <michele.polese@gmail.com>
 */
 
#ifndef RIC_CONTROL_MESSAGE_H
#define RIC_CONTROL_MESSAGE_H

#include "ns3/object.h"
#include <ns3/asn1c-types.h>

extern "C" {
  #include "E2AP-PDU.h"
  #include "E2SM-RC-ControlHeader.h"
  #include "E2SM-RC-ControlMessage.h"
  #include "E2SM-RC-ControlHeader-Format1.h"
  #include "E2SM-RC-ControlMessage-Format1.h"
  #include "RICcontrolRequest.h"
  #include "ProtocolIE-Field.h"
  #include "InitiatingMessage.h"
  #include "CellGlobalID.h"
  #include "NRCGI.h"
 }

namespace ns3 {

  class RicControlMessage : public SimpleRefCount<RicControlMessage>
  {
  public:
    enum ControlMessageServiceStyle {Radio_Bearer_Control=1,Radio_Resource_Allocation_Control=2, Connected_Mode_Mobility = 3, Energy_state = 300, Antenna_Control = 400};
    enum Connected_Mode_Mobility_Control_Action_ID {Handover_Control=1,Conditional_Handover_Control=2,DAPS_Handover_Control=3};
    enum Antenna_Control_Action_ID {RET_Tilt_Control=1};

    // Energy-state control action IDs (style 300); the encoder
    // (flexric/examples/xApp/c/orange/xapp_es_with_cell_util.c) sends these
    // as the ASCII digit characters '0'/'1'/'2' in the RIC Control Action ID
    // field, so the enum values below must match byte-for-byte.
    enum Energy_State_Control_Action_ID {Cell_Off='0', Cell_On='1', Cell_Sleep='2'};

    // custom RAN parameter ids used by the RET control message (style 400);
    // must match the encoder in flexric/examples/xApp/c/ctrl/xapp_ret_ctrl.c
    static constexpr uint32_t RET_PARAM_CONFIG_STRUCT = 32000;
    static constexpr uint32_t RET_PARAM_TILT_DEG = 32001;
    static constexpr uint32_t RET_PARAM_BEARING_DEG = 32002;

    /** Decoded RET (Remote Electrical Tilt) control action, style 400/action 1 */
    struct RetControl
    {
      bool valid{false};
      uint16_t cellId{0};
      double tiltDeg{0.0};
      bool hasBearing{false};
      double bearingDeg{0.0};
    };

    RicControlMessage (E2AP_PDU_t *pdu);
    ~RicControlMessage ();

   
    ControlMessageServiceStyle m_requestType;
    static std::vector<RANParameterItem> ExtractRANParametersFromControlMessage (
      E2SM_RC_ControlMessage_Format1_t *e2SmRcControlMessageFormat1);
    
    std::vector<RANParameterItem> m_valuesExtracted;
    RANfunctionID_t m_ranFunctionId;
    RICrequestID_t m_ricRequestId;
    RICcallProcessID_t m_ricCallProcessId;
    E2SM_RC_ControlHeader_Format1_t *m_e2SmRcControlHeaderFormat1;
    E2SM_RC_ControlMessage_Format1 *m_e2SmRcControlMessageFormat1;
    std::string GetSecondaryCellIdHO ();
    uint16_t GetTargetCell() const;
    RetControl GetRetControl() const;
  private:
    /**
    * Decodes the RIC Control message .
    *
    * \param pdu PDU passed by the RIC
    */
    void DecodeRicControlMessage (E2AP_PDU_t *pdu);
    std::string m_secondaryCellId;
  };
}

#endif /* RIC_CONTROL_MESSAGE_H */
